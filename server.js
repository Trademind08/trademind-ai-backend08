const express = require('express');
const cors = require('cors');
const multer = require('multer');
const OpenAI = require('openai');
require('dotenv').config();

const app = express();

app.use(cors());
app.use(express.json({ limit: '25mb' }));

const upload = multer({ storage: multer.memoryStorage() });

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function badValue(value) {
  if (!value) return true;
  const text = String(value).toLowerCase().trim();
  return (
    text === '' ||
    text.includes('sin') ||
    text.includes('no definido') ||
    text.includes('n/a') ||
    text.includes('null') ||
    text.includes('undefined')
  );
}

function normalizeSignal(value) {
  const text = String(value || 'NEUTRAL').toUpperCase();
  if (text.includes('BUY')) return 'BUY';
  if (text.includes('SELL')) return 'SELL';
  return 'NEUTRAL';
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    const match = String(text).match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch (_) {
        return {};
      }
    }
    return {};
  }
}

function forceAnalysis(input = {}) {
  const analysis = input && typeof input === 'object' ? input : {};

  const signal = normalizeSignal(analysis.signal);

  return {
    signal,

    confidence: badValue(analysis.confidence)
      ? 'Media'
      : analysis.confidence,

    entry_zone: badValue(analysis.entry_zone)
      ? signal === 'BUY'
        ? 'Entrada BUY condicional: esperar ruptura, rechazo o retesteo confirmado en zona clave.'
        : signal === 'SELL'
        ? 'Entrada SELL condicional: esperar ruptura, rechazo o retesteo confirmado en zona clave.'
        : 'NO TRADE: esperar confirmación clara antes de entrar.'
      : analysis.entry_zone,

    stop_loss: badValue(analysis.stop_loss)
      ? 'Stop técnico detrás de la zona de invalidación más cercana.'
      : analysis.stop_loss,

    take_profit_1: badValue(analysis.take_profit_1)
      ? 'TP1 en la primera zona lógica de soporte/resistencia a favor.'
      : analysis.take_profit_1,

    take_profit_2: badValue(analysis.take_profit_2)
      ? 'TP2 en la siguiente zona de liquidez o extensión.'
      : analysis.take_profit_2,

    invalidation_zone: badValue(analysis.invalidation_zone)
      ? 'Invalidación si el precio rompe en contra de la estructura actual.'
      : analysis.invalidation_zone,

    confirmation: badValue(analysis.confirmation)
      ? 'Esperar confirmación clara con rechazo, ruptura válida o retesteo antes de entrar.'
      : analysis.confirmation,

    no_trade_condition: badValue(analysis.no_trade_condition)
      ? 'No operar si el precio está lateral, sin volumen, lejos de la zona o sin confirmación clara.'
      : analysis.no_trade_condition,

    reading: badValue(analysis.reading)
      ? 'Lectura técnica basada en estructura, tendencia, liquidez, soporte/resistencia e invalidación.'
      : analysis.reading,

    risk_note: badValue(analysis.risk_note)
      ? 'Operación condicional. No entrar sin confirmación. Usar gestión de riesgo estricta.'
      : analysis.risk_note,

    ai_score: Number.isFinite(Number(analysis.ai_score))
      ? Number(analysis.ai_score)
      : 50,

    setup_quality: badValue(analysis.setup_quality)
      ? 'C'
      : analysis.setup_quality,

    risk_level: badValue(analysis.risk_level)
      ? 'Alto'
      : analysis.risk_level,

    market_context: badValue(analysis.market_context)
      ? 'Contexto evaluado con estructura, tendencia, liquidez, volatilidad y zonas técnicas visibles.'
      : analysis.market_context,

    trend: badValue(analysis.trend)
      ? 'Tendencia pendiente de confirmación por estructura.'
      : analysis.trend,

    liquidity_reading: badValue(analysis.liquidity_reading)
      ? 'Liquidez evaluada en máximos, mínimos, soportes, resistencias y zonas de reacción visibles.'
      : analysis.liquidity_reading,

    execution_trigger: badValue(analysis.execution_trigger)
      ? 'Ejecutar solo después de confirmación clara con rechazo, ruptura válida o retesteo.'
      : analysis.execution_trigger,

    missing_confirmation: badValue(analysis.missing_confirmation)
      ? 'Falta confirmación limpia de dirección, volumen o reacción estructural.'
      : analysis.missing_confirmation,

    institutional_summary: badValue(analysis.institutional_summary)
      ? 'El setup debe tratarse como condicional hasta que el precio confirme dirección e invalidación.'
      : analysis.institutional_summary,
  };
}

function buildInstitutionalPrompt() {
  return `
Eres TradeMind AI, un motor institucional de análisis técnico para traders de futuros, forex e índices.

Tu trabajo es analizar el gráfico enviado y devolver un PLAN DE TRADING profesional, accionable, prudente y estructurado.

REGLAS ABSOLUTAS:
1. Responde SOLO en JSON válido.
2. No escribas texto fuera del JSON.
3. No dejes campos vacíos.
4. Prohibido responder: "sin definir", "sin confirmar", "no definido", "n/a", "null" o frases vagas.
5. Si el gráfico no tiene claridad suficiente, devuelve NEUTRAL o NO TRADE con condiciones claras.
6. No inventes certeza. Usa escenarios condicionales cuando sea necesario.
7. Siempre entrega entrada, stop loss, take profit 1, take profit 2, invalidación, confirmación y condición de no operar.
8. El stop loss debe estar detrás de la invalidación técnica.
9. TP1 debe estar en la primera zona lógica a favor.
10. TP2 debe estar en la siguiente zona de liquidez o extensión.
11. Evalúa estructura, tendencia, liquidez, momentum, volatilidad, soporte/resistencia, order blocks, sweep, displacement y confirmación.
12. Si el setup no es limpio, dilo claramente como NO TRADE.
13. El lenguaje debe ser profesional, directo e institucional.

DEVUELVE SOLO ESTE JSON:

{
  "signal": "BUY | SELL | NEUTRAL",
  "confidence": "Alta | Media | Baja",
  "ai_score": 0,
  "setup_quality": "A | B | C | NO TRADE",
  "risk_level": "Bajo | Medio | Alto",
  "market_context": "Contexto general del mercado",
  "trend": "Lectura de tendencia",
  "liquidity_reading": "Lectura de liquidez",
  "entry_zone": "Zona o condición exacta de entrada",
  "stop_loss": "Nivel o zona exacta de stop loss",
  "take_profit_1": "Primer objetivo técnico",
  "take_profit_2": "Segundo objetivo técnico",
  "invalidation_zone": "Zona donde la idea queda invalidada",
  "confirmation": "Qué debe pasar antes de entrar",
  "execution_trigger": "Trigger exacto de ejecución",
  "missing_confirmation": "Qué confirmación falta si aún no es entrada válida",
  "no_trade_condition": "Cuándo NO operar",
  "reading": "Lectura técnica breve y profesional",
  "institutional_summary": "Resumen institucional final",
  "risk_note": "Nota clara de gestión de riesgo"
}
`;
}

app.get('/', (req, res) => {
  res.json({
    ok: true,
    message: 'TradeMind AI Backend Running',
    engine: 'Institutional Analysis Engine',
  });
});

app.post('/analyze-chart', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.json({
        ok: false,
        error: 'No se recibió imagen.',
        analysis: forceAnalysis({}),
      });
    }

    const base64Image = req.file.buffer.toString('base64');

    const response = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: buildInstitutionalPrompt(),
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analiza este gráfico y devuelve únicamente JSON válido.',
            },
            {
              type: 'image_url',
              image_url: {
                url: `data:image/jpeg;base64,${base64Image}`,
              },
            },
          ],
        },
      ],
      temperature: 0.2,
      max_tokens: 1200,
    });

    const text = response.choices?.[0]?.message?.content || '';
    const parsed = safeJsonParse(text);
    const finalAnalysis = forceAnalysis(parsed);

    return res.json({
      ok: true,
      engine: 'institutional',
      analysis: finalAnalysis,
    });
  } catch (error) {
    console.error('Error en /analyze-chart:', error);

    return res.json({
      ok: false,
      error: 'Error en análisis',
      analysis: forceAnalysis({}),
    });
  }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, '0.0.0.0', () => {
  console.log(`TradeMind AI Institutional Engine corriendo en puerto ${PORT}`);
});
