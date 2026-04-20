const express = require('express');
const router = express.Router();
const { GoogleGenerativeAI } = require('@google/generative-ai');
const authMiddleware = require('../middleware/auth');

router.use(authMiddleware);

function buildPrompt(message, conversationContext = [], conversationMemory = '') {
  const contextLines = renderConversationContext(conversationContext);
  return [
    'Eres un asistente de apoyo emocional avanzado para una app de ayuda psicológica.',
    'Analiza el mensaje del usuario y responde SOLO con JSON válido, sin markdown ni texto adicional.',
    'Formato exacto:',
    '{"category":"crisis|ansiedad|estres|tristeza|familiar|sueno|general","label":"texto corto","reply":"respuesta empatica en español","recommendedSpecialty":"texto corto","crisis":true|false,"emotions":{"primary":"alegria|tristeza|ira|miedo|ansiedad|culpa|vergüenza|soledad|esperanza|frustracion|confianza|desesperanza","intensity":0.1-1.0,"secondary":"emocion opcional"},"therapies":[{"name":"CBT|DBT|EMDR|ACT|MBCT|etc","description":"breve descripcion","suitability":0.1-1.0}],"metrics":{"confidence":0.1-1.0,"processingTime":0,"sentimentScore":-1.0-1.0}}',
    'Reglas de análisis de sentimientos:',
    '- emotions.primary: emoción principal detectada (alegria, tristeza, ira, miedo, ansiedad, culpa, vergüenza, soledad, esperanza, frustracion, confianza, desesperanza)',
    '- emotions.intensity: intensidad de la emoción (0.1=baja, 1.0=muy alta)',
    '- emotions.secondary: emoción secundaria opcional si aplica',
    '- metrics.sentimentScore: puntuación de sentimiento general (-1.0=muy negativo, 1.0=muy positivo)',
    '- metrics.confidence: confianza en el análisis (0.1-1.0)',
    '- metrics.processingTime: tiempo estimado de procesamiento en ms',
    '',
    'Reglas de terapias específicas:',
    '- therapies: array de terapias recomendadas (máximo 3)',
    '- Cada terapia debe incluir: name, description, suitability (0.1-1.0)',
    '- Terapias disponibles: CBT (Terapia Cognitivo-Conductual), DBT (Terapia Dialéctica Conductual), EMDR (Desensibilización y Reprocesamiento por Movimientos Oculares), ACT (Terapia de Aceptación y Compromiso), MBCT (Terapia Cognitiva Basada en Mindfulness), IPT (Terapia Interpersonal), EFT (Terapia Centrada en Emociones), Gestalt, Psicodinámica, Humanista',
    '- suitability: qué tan adecuada es la terapia para este caso específico',
    '',
    'Reglas generales:',
    '- Si detectas riesgo de autolesión, suicidio o peligro inmediato, category debe ser crisis y crisis=true.',
    '- reply debe sonar como una persona que acompaña, no como una plantilla.',
    '- reply debe ser breve, empática, concreta y en español natural.',
    '- No repitas ni paraphrasees literalmente el mensaje del usuario.',
    '- Usa el contexto reciente para responder con criterio y continuidad.',
    '- recommendedSpecialty debe ser una especialidad de psicología útil.',
    '- memory debe incluir hechos estables: nombre, profesión, hobbies, contexto familiar, riesgos.',
    '',
    conversationMemory ? `Memoria persistente actual:\n${conversationMemory}` : 'Memoria persistente actual: vacía.',
    '',
    contextLines ? `Contexto reciente:\n${contextLines}` : 'Contexto reciente: sin contexto adicional.',
    '',
    `Mensaje del usuario: ${message}`,
  ].join('\n');
}

function renderConversationContext(conversationContext) {
  if (!Array.isArray(conversationContext) || conversationContext.length === 0) {
    return '';
  }

  return conversationContext
    .slice(-6)
    .map((entry) => {
      if (entry && typeof entry === 'object') {
        const role = String(entry.role || 'unknown').toLowerCase();
        const text = String(entry.text || '').trim();
        if (!text) return '';
        return `${role}: ${text}`;
      }

      const text = String(entry || '').trim();
      return text ? `message: ${text}` : '';
    })
    .filter(Boolean)
    .join('\n');
}

function safeParseJson(text) {
  const trimmed = String(text || '').trim();
  const fenced = trimmed
    .replace(/^```(?:json)?/i, '')
    .replace(/```$/i, '')
    .trim();

  const start = fenced.indexOf('{');
  const end = fenced.lastIndexOf('}');
  const cleaned = start >= 0 && end > start ? fenced.slice(start, end + 1) : fenced;

  return JSON.parse(cleaned);
}

function normalizeText(text) {
  return String(text || '')
    .toLowerCase()
    .trim()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9áéíóúüñ\s]/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function isTooSimilarReply(reply, message) {
  const normalizedReply = normalizeText(reply);
  const normalizedMessage = normalizeText(message);

  if (!normalizedReply || !normalizedMessage) return false;
  if (normalizedReply.includes(normalizedMessage) || normalizedMessage.includes(normalizedReply)) {
    return true;
  }

  const replyWords = new Set(normalizedReply.split(' ').filter((word) => word.length > 2));
  const messageWords = new Set(normalizedMessage.split(' ').filter((word) => word.length > 2));

  if (replyWords.size === 0 || messageWords.size === 0) return false;

  let overlap = 0;
  messageWords.forEach((word) => {
    if (replyWords.has(word)) overlap += 1;
  });

  return overlap / messageWords.size >= 0.6;
}

function isMemoryRecallIntent(message) {
  const normalized = normalizeText(message);
  return (
    normalized.includes('te acuerdas') ||
    normalized.includes('recuerdas') ||
    normalized.includes('recordas') ||
    normalized.includes('hablabamos') ||
    normalized.includes('hablamos') ||
    normalized.includes('del tema') ||
    normalized.includes('lo que te dije')
  );
}

function hasUsableContext(conversationContext = [], conversationMemory = '') {
  const memory = String(conversationMemory || '').trim();
  if (memory.length >= 12) return true;
  if (!Array.isArray(conversationContext)) return false;
  return conversationContext.some((entry) => {
    const text = String((entry && entry.text) || '').trim();
    return text.length >= 8;
  });
}

function isGenericReplyText(reply) {
  const normalized = normalizeText(reply);
  if (!normalized) return true;
  const genericPatterns = [
    'te leo',
    'cuentame un poco mas',
    'cuentame que te trae por aca',
    'te respondo con algo mas util',
    'paso a paso',
  ];
  return genericPatterns.some((pattern) => normalized.includes(pattern));
}

function isSummaryIntent(message) {
  const normalized = normalizeText(message);
  return (
    normalized.includes('resumen') ||
    normalized.includes('resumir') ||
    normalized.includes('resumeme') ||
    normalized.includes('resumeme') ||
    normalized.includes('resumeme') ||
    normalized.includes('hazme un resumen') ||
    normalized.includes('resume lo que') ||
    normalized.includes('que te conte') ||
    normalized.includes('que te dije')
  );
}

function isOutOfScopeIntent(message) {
  const normalized = normalizeText(message);
  const psychSignals = [
    'emocion',
    'emocional',
    'sentir',
    'siento',
    'ansiedad',
    'estres',
    'triste',
    'deprim',
    'llorar',
    'familia',
    'padre',
    'madre',
    'mama',
    'papa',
    'pareja',
    'hijo',
    'hija',
    'hermano',
    'hermana',
    'trauma',
    'traumas',
    'abuso',
    'violencia',
    'crisis',
    'suicid',
    'dormir',
    'sueno',
    'sueño',
  ];
  const offTopicSignals = [
    'jugar',
    'juego',
    'juega',
    'free fire',
    'free firee',
    'freefire',
    'colombia juega',
    'que dia juega',
    'partido',
    'futbol',
    'baloncesto',
    'tenis',
    'formula 1',
    'clima',
    'temperatura',
    'noticias',
    'politica',
    'politico',
    'musica',
    'cancion',
    'pelicula',
    'serie',
    'videojuego',
    'gaming',
    'codigo',
    'programar',
    'matematic',
    'tarea',
    'examen',
    'escuela',
    'colegio',
    'trabajo',
  ];

  if (psychSignals.some((signal) => normalized.includes(signal))) {
    return false;
  }

  return offTopicSignals.some((signal) => normalized.includes(signal));
}

function buildOutOfScopeReply() {
  return pickVariant([
    'Oye, solo estoy para ayudarte con temas psicológicos, traumas y temas familiares. Si quieres, cuéntame cómo te sientes o qué te preocupa y ahí sí te acompaño.',
    'Ese tema se sale de mi enfoque. Yo te apoyo en temas psicológicos, traumas y familia. Si quieres, dime cómo te estás sintiendo y lo vemos juntos.',
    'No soy para deportes o temas generales; estoy para acompañarte en lo emocional, traumas y familia. Si quieres, seguimos por ahí.',
  ], 'out_of_scope');
}

function pickVariant(options, seed = '') {
  if (!Array.isArray(options) || options.length === 0) return '';
  const normalizedSeed = normalizeText(seed);
  const hash = Array.from(normalizedSeed).reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return options[hash % options.length];
}

function summarizeMemoryLines(conversationMemory = '') {
  const lines = String(conversationMemory || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.toLowerCase().startsWith('perfil:'));

  return lines.slice(0, 3).join('; ');
}

function summarizeRecentUserContext(conversationContext = []) {
  if (!Array.isArray(conversationContext)) return '';

  const userMessages = conversationContext
    .filter((entry) => {
      const role = String((entry && entry.role) || '').toLowerCase();
      return role === 'user';
    })
    .map((entry) => String((entry && entry.text) || '').trim())
    .filter(Boolean);

  const filteredMessages = userMessages.filter((msg) => !isSummaryIntent(msg));

  if (filteredMessages.length === 0) return '';

  return filteredMessages
    .slice(-2)
    .map((msg) => msg.length > 100 ? `${msg.slice(0, 100)}...` : msg)
    .join(' | ');
}

function buildSummaryReply(conversationContext = [], conversationMemory = '') {
  const memoryLines = String(conversationMemory || '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .filter((line) => !line.toLowerCase().startsWith('perfil:'));

  const hasUsefulMemory = memoryLines.some((line) => !line.toLowerCase().startsWith('ultimo tema:'));
  const cleanMemoryLines = hasUsefulMemory
    ? memoryLines.filter((line) => !line.toLowerCase().startsWith('ultimo tema:'))
    : memoryLines;

  const memorySummary = cleanMemoryLines.slice(0, 3).join('; ');
  const contextSummary = summarizeRecentUserContext(conversationContext);
  const combined = `${memorySummary.toLowerCase()} ${contextSummary.toLowerCase()}`;
  const hasFamilyContext =
    combined.includes('familiar') ||
    combined.includes('mama') ||
    combined.includes('papá') ||
    combined.includes('papa') ||
    combined.includes('casa');
  const hasViolenceContext =
    combined.includes('violencia') ||
    combined.includes('maltrato') ||
    combined.includes('golpe') ||
    combined.includes('agred');
  const hasNightContext = combined.includes('noche');
  const intro = (hasFamilyContext || hasViolenceContext)
    ? 'Sí, me acuerdo. Sé que este tema te duele.'
    : 'Sí, me acuerdo de lo que venimos hablando.';

  if (!memorySummary && !contextSummary) {
    return 'Puedo resumirte mejor si me dices de cuál parte quieres el resumen: lo familiar, cómo te has sentido, o lo último que hablamos .';
  }

  if (memorySummary && contextSummary) {
    const followUp = hasNightContext
      ? 'Si quieres, retomemos desde esa noche: ¿qué fue lo más difícil para ti en ese momento?'
      : 'Si quieres, retomemos por la parte que más te pesa ahora para ayudarte con algo concreto.';
    return `${intro} En resumen, me contaste: ${memorySummary}. Lo último que mencionaste fue: ${contextSummary}. ${followUp}`;
  }

  const base = memorySummary || contextSummary;
  return `${intro} En resumen, me contaste: ${base}. Si quieres, lo bajamos a un siguiente paso concreto para hoy.`;
}

function buildFallbackReply(category, message, conversationContext = [], conversationMemory = '') {
  const normalized = normalizeText(message);
  if (isOutOfScopeIntent(message)) {
    return buildOutOfScopeReply();
  }

  const wantsHumanSupport =
    normalized.includes('quiero hablar con una persona') ||
    normalized.includes('quiero hablar con alguien') ||
    normalized.includes('quiero hablar con un psicologo') ||
    normalized.includes('quiero hablar con psicologo') ||
    normalized.includes('quiero hablar con una psicologa') ||
    normalized.includes('quiero hablar con psicologa') ||
    normalized.includes('con una persona') ||
    normalized.includes('con alguien') ||
    normalized.includes('pasame con una persona') ||
    normalized.includes('pasame con alguien') ||
    normalized.includes('pasame con el psicologo') ||
    normalized.includes('pasame con la psicologa') ||
    normalized.includes('derivame con un psicologo') ||
    normalized.includes('derivame con una persona') ||
    normalized.includes('necesito una persona') ||
    normalized.includes('necesito hablar con alguien');
  const hasViolenceInHome =
    (normalized.includes('padre') && normalized.includes('madre') &&
      (normalized.includes('pega') || normalized.includes('golpea') || normalized.includes('maltrata') || normalized.includes('agrede'))) ||
    (normalized.includes('papá') && normalized.includes('mamá') &&
      (normalized.includes('pega') || normalized.includes('golpea') || normalized.includes('maltrata') || normalized.includes('agrede')));

  if (
    normalized.includes('me pegaron') ||
    normalized.includes('me golpearon') ||
    normalized.includes('me pego') ||
    normalized.includes('me golpeo') ||
    normalized.includes('me maltratan') ||
    normalized.includes('me maltrataron') ||
    normalized.includes('violencia') ||
    normalized.includes('abuso') ||
    normalized.includes('me agredieron') ||
    hasViolenceInHome
  ) {
    return 'Lo que cuentas es serio. Si ahora mismo hay golpes o riesgo en casa, busca a un adulto de confianza, sal a un lugar seguro si puedes y pide ayuda de inmediato. ¿Están a salvo ahora mismo?';
  }

  if (wantsHumanSupport) {
    return 'Claro, te puedo orientar con un psicólogo. Si quieres, te ayudo a seguir por aquí y a la vez te conecto con apoyo humano para que no tengas que cargarlo solo/a.';
  }

  if (isSummaryIntent(message)) {
    return buildSummaryReply(conversationContext, conversationMemory);
  }

  if (isMemoryRecallIntent(message) && hasUsableContext(conversationContext, conversationMemory)) {
    return 'Sí, me acuerdo de lo que veníamos hablando. Si quieres, retomemos desde lo último que te estaba pesando para ayudarte con algo concreto ahora mismo.';
  }

  if (category === 'crisis') {
    return 'Lo que cuentas me preocupa. Si hay peligro ahora mismo, busca a una persona de confianza o emergencias ya mismo; si quieres, me quedo contigo para pensar el siguiente paso.';
  }

  if (category === 'ansiedad') {
    return 'Eso suena muy cargado por dentro. Cuéntame en qué momentos se pone peor y qué notas en tu cuerpo para ubicarlo mejor.';
  }

  if (category === 'estres') {
    return 'Parece que traes demasiado encima. Si me dices qué es lo que más te está agotando hoy, lo ordenamos juntos sin prisa.';
  }

  if (category === 'tristeza') {
    return 'Se siente pesado lo que estás contando. Si quieres, dime qué fue lo que más te golpeó hoy y lo vemos paso a paso.';
  }

  if (category === 'familiar') {
    if (hasViolenceInHome) {
      return 'Lo que cuentas es serio. Si ahora mismo hay golpes o riesgo en casa, busca a un adulto de confianza, sal a un lugar seguro si puedes y pide ayuda de inmediato. ¿Tu mamá está a salvo ahora mismo?';
    }

    return 'Lo que pasa en tu casa suena duro. Si quieres, dime qué ocurrió primero y qué es lo que más te preocupa ahora para ayudarte a ordenar el siguiente paso.';
  }

  if (category === 'sueno') {
    return 'Dormir mal te puede dejar todo más cuesta arriba. Dime si te cuesta dormirte, te despiertas mucho o te levantas sin energía, y te respondo más fino.';
  }

  if (normalized.includes('hola') || normalized.includes('buenas') || normalized.includes('buenos dias')) {
    if (hasUsableContext(conversationContext, conversationMemory)) {
      return 'Hola. Te sigo el hilo; si quieres retomamos lo último que venías trabajando y vemos qué cambió hoy.';
    }
    return 'Cuéntame qué te trae por acá hoy y voy contigo paso a paso. :)';
  }

  if (normalized.includes('mi papa') || normalized.includes('mi papá') || normalized.includes('mi mama') || normalized.includes('mi mamá') || normalized.includes('mis papás') || normalized.includes('mis papas')) {
    if (hasViolenceInHome) {
      return 'Lo que cuentas es serio. Si ahora mismo hay golpes o riesgo en casa, busca a un adulto de confianza, sal a un lugar seguro si puedes y pide ayuda de inmediato. ¿Tu mamá está a salvo ahora mismo?';
    }

    return 'Lo que pasa en tu casa suena duro. Si quieres, dime qué ocurrió primero y qué es lo que más te pesa ahora para ayudarte a ordenar el siguiente paso.';
  }

  if (wantsHumanSupport) {
    return 'Claro, te acompaño y te conecto con apoyo humano. Si quieres, seguimos por aquí mientras preparo la derivación.';
  }

  return pickVariant([
    'Te leo. Cuéntame un poco más de lo que pasó y te respondo con algo más útil y directo.',
    'Estoy contigo. Si me das un poco más de contexto, te respondo de forma más concreta.',
    'Gracias por abrir el tema. Cuéntame un poco más para ayudarte con algo realmente útil.',
  ], `${category}|${message}`);
}

function detectLocalCategory(message) {
  const normalized = normalizeText(message);

  if (
    normalized.includes('suicid') ||
    normalized.includes('matarme') ||
    normalized.includes('hacerme daño') ||
    normalized.includes('hacerme dano') ||
    normalized.includes('golpe') ||
    normalized.includes('violencia') ||
    normalized.includes('abuso')
  ) {
    return 'crisis';
  }

  if (
    normalized.includes('ansiedad') ||
    normalized.includes('nervios') ||
    normalized.includes('pánico') ||
    normalized.includes('panico')
  ) {
    return 'ansiedad';
  }

  if (
    normalized.includes('estres') ||
    normalized.includes('estres') ||
    normalized.includes('agotado') ||
    normalized.includes('presion') ||
    normalized.includes('presión')
  ) {
    return 'estres';
  }

  if (normalized.includes('triste') || normalized.includes('deprim') || normalized.includes('llorar')) {
    return 'tristeza';
  }

  if (
    normalized.includes('familia') ||
    normalized.includes('casa') ||
    normalized.includes('papá') ||
    normalized.includes('papa') ||
    normalized.includes('mamá') ||
    normalized.includes('mama')
  ) {
    return 'familiar';
  }

  if (
    normalized.includes('dorm') ||
    normalized.includes('sueño') ||
    normalized.includes('sueno') ||
    normalized.includes('insomnio')
  ) {
    return 'sueno';
  }

  return 'general';
}

router.post('/triage', async (req, res) => {
  const startTime = Date.now();
  try {
    const { message, conversationContext, conversationMemory } = req.body;
    if (!message || !String(message).trim()) {
      return res.status(400).json({ error: 'Falta el mensaje' });
    }

    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'GEMINI_API_KEY no está configurada' });
    }

    if (isOutOfScopeIntent(message)) {
      const processingTime = Date.now() - startTime;
      return res.json({
        category: 'general',
        label: 'fuera de alcance',
        reply: buildOutOfScopeReply(),
        recommendedSpecialty: 'Bienestar emocional',
        crisis: false,
        memory: String(conversationMemory || '').trim(),
        emotions: {
          primary: 'neutral',
          intensity: 0.3,
          secondary: null
        },
        therapies: [],
        metrics: {
          confidence: 0.9,
          processingTime,
          sentimentScore: 0.0
        },
        raw: { fallback: true, reason: 'Tema fuera de alcance' },
      });
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const model = genAI.getGenerativeModel({ 
  model: 'gemini-2.5-flash' 
});
    const incomingMemory = String(conversationMemory || '').trim();
    const result = await model.generateContent(
      buildPrompt(
        String(message),
        Array.isArray(conversationContext) ? conversationContext : [],
        incomingMemory,
      ),
    );
    const text = result.response.text();

    let parsed;
    try {
      parsed = safeParseJson(text);
    } catch (_) {
      parsed = null;
    }

    const processingTime = Date.now() - startTime;

    if (!parsed || typeof parsed !== 'object') {
      const localCategory = detectLocalCategory(message);
      return res.json({
        category: localCategory,
        label: localCategory === 'general' ? 'orientación general' : localCategory,
          reply: buildFallbackReply(
            localCategory,
            String(message),
            Array.isArray(conversationContext) ? conversationContext : [],
            incomingMemory,
          ),
        recommendedSpecialty: localCategory === 'crisis' ? 'Crisis y contención' : 'Bienestar emocional',
        crisis: localCategory === 'crisis',
        memory: incomingMemory,
        emotions: {
          primary: localCategory === 'crisis' ? 'miedo' : 'neutral',
          intensity: localCategory === 'crisis' ? 0.8 : 0.3,
          secondary: null
        },
        therapies: localCategory === 'crisis' ? 
          [{ name: 'DBT', description: 'Terapia Dialéctica Conductual para manejo de crisis', suitability: 0.8 }] : 
          [{ name: 'CBT', description: 'Terapia Cognitivo-Conductual para bienestar general', suitability: 0.6 }],
        metrics: {
          confidence: 0.5,
          processingTime,
          sentimentScore: localCategory === 'crisis' ? -0.7 : 0.0
        },
        raw: { fallback: true, reason: 'Gemini no devolvió JSON válido' },
      });
    }

    const category = String(parsed.category || 'general').trim();
    const label = String(parsed.label || 'orientación general').trim();
    let reply = String(parsed.reply || '').trim();
    const recommendedSpecialty = String(parsed.recommendedSpecialty || 'Bienestar emocional').trim();
    const memory = String(parsed.memory || incomingMemory).trim();
    const crisisValue = parsed.crisis;
    const crisis = crisisValue === true || crisisValue === 'true' || crisisValue === 1 || crisisValue === '1';
    const summaryIntent = isSummaryIntent(String(message));

    // Análisis de sentimientos con valores por defecto
    const emotions = parsed.emotions || {};
    const emotionsResult = {
      primary: String(emotions.primary || 'neutral').trim(),
      intensity: Math.max(0.1, Math.min(1.0, parseFloat(emotions.intensity) || 0.5)),
      secondary: emotions.secondary ? String(emotions.secondary).trim() : null
    };

    // Terapias recomendadas con validación
    const therapies = Array.isArray(parsed.therapies) ? parsed.therapies.slice(0, 3).map(therapy => ({
      name: String(therapy.name || 'CBT').trim(),
      description: String(therapy.description || 'Terapia recomendada').trim(),
      suitability: Math.max(0.1, Math.min(1.0, parseFloat(therapy.suitability) || 0.5))
    })) : [{ name: 'CBT', description: 'Terapia Cognitivo-Conductual', suitability: 0.6 }];

    // Métricas de efectividad
    const metrics = parsed.metrics || {};
    const metricsResult = {
      confidence: Math.max(0.1, Math.min(1.0, parseFloat(metrics.confidence) || 0.7)),
      processingTime,
      sentimentScore: Math.max(-1.0, Math.min(1.0, parseFloat(metrics.sentimentScore) || 0.0))
    };

    if (summaryIntent) {
      reply = buildSummaryReply(
        Array.isArray(conversationContext) ? conversationContext : [],
        incomingMemory,
      );
    }

    if (!reply || isTooSimilarReply(reply, String(message)) || isGenericReplyText(reply)) {
      reply = buildFallbackReply(
        category,
        String(message),
        Array.isArray(conversationContext) ? conversationContext : [],
        incomingMemory,
      );
    }

    return res.json({
      category,
      label,
      reply,
      recommendedSpecialty,
      crisis,
      memory,
      emotions: emotionsResult,
      therapies,
      metrics: metricsResult,
      raw: parsed,
    });
  } catch (error) {
    const processingTime = Date.now() - startTime;
    console.error('Error en triage:', error);
    return res.status(500).json({ 
      error: 'Error interno del servidor',
      metrics: {
        confidence: 0.0,
        processingTime,
        sentimentScore: 0.0
      }
    });
  }
});
    const message = String(req.body?.message || '');
    const conversationContext = Array.isArray(req.body?.conversationContext)
      ? req.body.conversationContext
      : [];
    const conversationMemory = String(req.body?.conversationMemory || '').trim();
    const localCategory = detectLocalCategory(message);
    console.error('[AI][TRIAGE][FALLBACK]', error.message);
    return res.json({
      category: localCategory,
      label: localCategory === 'general' ? 'orientación general' : localCategory,
      reply: buildFallbackReply(localCategory, message, conversationContext, conversationMemory),
      recommendedSpecialty: localCategory === 'crisis' ? 'Crisis y contención' : 'Bienestar emocional',
      crisis: localCategory === 'crisis',
      memory: conversationMemory,
      raw: { fallback: true, error: error.message },
    });

// Ruta temporal para listar modelos disponibles de Gemini
router.get('/list-gemini-models', async (req, res) => {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: 'GEMINI_API_KEY no está configurada' });
    }
    const genAI = new GoogleGenerativeAI(apiKey);
    const models = await genAI.listModels();
    res.json(models);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
