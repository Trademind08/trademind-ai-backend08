import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";

const app = express();

app.use(cors());
app.use(express.json({ limit: "30mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 12 * 1024 * 1024,
  },
});

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const PORT = process.env.PORT || 10000;
const ENGINE_VERSION = "trademind-institutional-final-v5";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";

function normalizeText(value) {
  return String(value || "").trim();
}

function badValue(value) {
  if (value === null || value === undefined) return true;

  const text = String(value).toLowerCase().trim();

  return (
    text === "" ||
    text === "." ||
    text === "-" ||
    text === "n/a" ||
    text === "na" ||
    text === "null" ||
    text === "undefined" ||
    text.includes("no aplicable") ||
    text.includes("sin definir") ||
    text.includes("sin confirmar") ||
    text.includes("no definido") ||
    text.includes("no hay entrada") ||
    text.includes("no se puede")
  );
}

function normalizeSignal(value) {
  const text = normalizeText(value).toUpperCase();

  if (text.includes("BUY") || text.includes("COMPRA") || text.includes("ALCISTA")) {
    return "BUY";
  }

  if (text.includes("SELL") || text.includes("VENTA") || text.includes("BAJISTA")) {
    return "SELL";
  }

  return "NEUTRAL";
}

function normalizeMarketType(value) {
  const text = normalizeText(value).toUpperCase();

  if (text.includes("FOREX") || text.includes("FX")) return "FOREX";
  if (text.includes("FUTURE")) return "FUTURES";
  if (text.includes("CRYPTO")) return "CRYPTO";
  if (text.includes("STOCK")) return "STOCKS";

  return "UNKNOWN";
}

function inferMarketTypeFromSymbol(symbol = "") {
  const s = normalizeText(symbol).toUpperCase().replace("/", "");

  const futures = ["MNQ", "NQ", "MES", "ES", "YM", "MYM", "RTY", "M2K", "CL", "GC", "MGC"];
  const forex = ["EURUSD", "GBPUSD", "USDJPY", "USDCAD", "USDCHF", "AUDUSD", "NZDUSD", "EURJPY", "GBPJPY", "XAUUSD", "XAGUSD"];
  const crypto = ["BTCUSD", "BTCUSDT", "ETHUSD", "ETHUSDT", "SOLUSD", "XRPUSD"];

  if (futures.includes(s)) return "FUTURES";
  if (forex.includes(s)) return "FOREX";
  if (crypto.includes(s)) return "CRYPTO";

  return "UNKNOWN";
}

function extractPrices(text) {
  if (!text) return [];
  const matches = String(text).match(/\d+(\.\d+)?/g);
  return matches ? matches.map(Number) : [];
}

function average(values) {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function clampScore(value) {
  const score = Number(value);
  if (Number.isNaN(score)) return 58;
  return Math.max(0, Math.min(100, Math.round(score)));
}

function getPriceDecimals(symbol = "", marketType = "", price = null) {
  const s = normalizeText(symbol).toUpperCase().replace("/", "");
  const type = normalizeMarketType(marketType);

  if (s.includes("JPY")) return 3;
  if (s === "XAUUSD" || s === "XAGUSD") return 2;
  if (type === "FOREX") return 5;
  if (price && price > 0 && price < 100) return 5;

  return 2;
}

function formatPrice(price, symbol = "", marketType = "") {
  if (price === null || price === undefined || Number.isNaN(Number(price))) {
    return null;
  }

  return Number(price).toFixed(getPriceDecimals(symbol, marketType, price));
}

function getDefaultBuffer(entryPrice, symbol = "", marketType = "") {
  const s = normalizeText(symbol).toUpperCase().replace("/", "");
  const type = normalizeMarketType(marketType);

  if (s === "XAUUSD") return 2.5;
  if (s.includes("JPY")) return 0.15;
  if (type === "FOREX") return 0.0012;
  if (type === "CRYPTO") return entryPrice ? entryPrice * 0.006 : 50;

  return 25;
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch (_) {
    const match = String(text || "").match(/\{[\s\S]*\}/);
    if (!match) return {};

    try {
      return JSON.parse(match[0]);
    } catch (_) {
      return {};
    }
  }
}

function normalizeConfidence(value, aiScore) {
  const text = normalizeText(value).toLowerCase();

  if (text.includes("alta") || text.includes("high")) return "Alta";
  if (text.includes("media") || text.includes("medium")) return "Media";
  if (text.includes("baja") || text.includes("low")) return "Baja";

  if (aiScore >= 76) return "Alta";
  if (aiScore >= 52) return "Media";

  return "Baja";
}

function normalizeSetupQuality(value, aiScore, signal) {
  const text = normalizeText(value).toUpperCase();

  if (["A", "B", "C", "NO TRADE"].includes(text)) return text;

  if (signal === "NEUTRAL") return "NO TRADE";
  if (aiScore >= 78) return "A";
  if (aiScore >= 62) return "B";

  return "C";
}

function normalizeRiskLevel(value, aiScore) {
  const text = normalizeText(value).toLowerCase();

  if (text.includes("bajo")) return "Bajo";
  if (text.includes("medio")) return "Medio";
  if (text.includes("alto")) return "Alto";

  if (aiScore >= 75) return "Medio";
  if (aiScore >= 55) return "Medio";

  return "Alto";
}

function forceDirectionalSignal(analysis = {}) {
  const current = normalizeSignal(analysis.signal);
  if (current !== "NEUTRAL") return current;

  const text = [
    analysis.market_context,
    analysis.trend,
    analysis.liquidity_reading,
    analysis.entry_zone,
    analysis.confirmation,
    analysis.execution_trigger,
    analysis.reading,
    analysis.institutional_summary,
  ]
    .join(" ")
    .toLowerCase();

  const bullishWords = [
    "alcista",
    "buy",
    "compra",
    "soporte",
    "demanda",
    "demand",
    "ruptura alcista",
    "rechazo alcista",
    "bos alcista",
    "choch alcista",
    "higher low",
  ];

  const bearishWords = [
    "bajista",
    "sell",
    "venta",
    "resistencia",
    "oferta",
    "supply",
    "ruptura bajista",
    "rechazo bajista",
    "bos bajista",
    "choch bajista",
    "lower high",
  ];

  const bull = bullishWords.filter((w) => text.includes(w)).length;
  const bear = bearishWords.filter((w) => text.includes(w)).length;

  if (bull >= bear + 1) return "BUY";
  if (bear >= bull + 1) return "SELL";

  return "NEUTRAL";
}

function forceAnalysis(input = {}, metadata = {}) {
  const analysis = input && typeof input === "object" ? input : {};

  const symbol = normalizeText(analysis.symbol || metadata.symbol || "UNKNOWN").toUpperCase();
  const requestedMarketType = normalizeMarketType(metadata.marketType);
  const modelMarketType = normalizeMarketType(analysis.market_type);
  const inferredMarketType = inferMarketTypeFromSymbol(symbol);

  const marketType =
    requestedMarketType !== "UNKNOWN"
      ? requestedMarketType
      : modelMarketType !== "UNKNOWN"
      ? modelMarketType
      : inferredMarketType;

  let signal = forceDirectionalSignal(analysis);
  let aiScore = clampScore(analysis.ai_score);

  if (signal !== "NEUTRAL" && aiScore < 58) {
    aiScore = 58;
  }

  const entryPrices = extractPrices(analysis.entry_zone);
  const stopPrices = extractPrices(analysis.stop_loss);
  const tp1Prices = extractPrices(analysis.take_profit_1);
  const tp2Prices = extractPrices(analysis.take_profit_2);
  const invalidationPrices = extractPrices(analysis.invalidation_zone);

  const entryPrice = average(entryPrices);

  let stopPrice = stopPrices.length ? stopPrices[0] : null;
  let tp1Price = tp1Prices.length ? tp1Prices[0] : null;
  let tp2Price = tp2Prices.length ? tp2Prices[0] : null;

  const invalidationPrice = invalidationPrices.length ? invalidationPrices[0] : null;

  if (!stopPrice && invalidationPrice) {
    stopPrice = invalidationPrice;
  }

  const buffer = getDefaultBuffer(entryPrice, symbol, marketType);

  if (!stopPrice && entryPrice && signal === "BUY") {
    stopPrice = entryPrice - buffer;
  }

  if (!stopPrice && entryPrice && signal === "SELL") {
    stopPrice = entryPrice + buffer;
  }

  const risk =
    entryPrice && stopPrice && signal !== "NEUTRAL"
      ? Math.abs(entryPrice - stopPrice)
      : null;

  if (!tp1Price && entryPrice && risk && signal === "BUY") {
    tp1Price = entryPrice + risk;
  }

  if (!tp2Price && entryPrice && risk && signal === "BUY") {
    tp2Price = entryPrice + risk * 2;
  }

  if (!tp1Price && entryPrice && risk && signal === "SELL") {
    tp1Price = entryPrice - risk;
  }

  if (!tp2Price && entryPrice && risk && signal === "SELL") {
    tp2Price = entryPrice - risk * 2;
  }

  const marketLabel =
    marketType === "FOREX"
      ? "forex"
      : marketType === "FUTURES"
      ? "futuros"
      : marketType === "CRYPTO"
      ? "crypto"
      : "mercado";

  const fallbackEntry =
    signal === "BUY"
      ? `BUY condicional en ${marketLabel}: esperar retroceso hacia zona de demanda/soporte o ruptura alcista con retesteo confirmado.`
      : signal === "SELL"
      ? `SELL condicional en ${marketLabel}: esperar retroceso hacia zona de oferta/resistencia o ruptura bajista con retesteo confirmado.`
      : `NO TRADE en ${marketLabel}: esperar ruptura clara, rechazo institucional o dirección definida antes de operar.`;

  return {
    market_type: marketType,
    symbol: symbol || "UNKNOWN",

    signal,

    confidence: normalizeConfidence(analysis.confidence, aiScore),
    ai_score: aiScore,
    setup_quality: normalizeSetupQuality(analysis.setup_quality, aiScore, signal),
    risk_level: normalizeRiskLevel(analysis.risk_level, aiScore),

    market_context: badValue(analysis.market_context)
      ? `Contexto ${marketLabel}: lectura basada en estructura visible, zonas de reacción, liquidez y dirección probable.`
      : String(analysis.market_context),

    trend: badValue(analysis.trend)
      ? "Tendencia evaluada por estructura reciente, máximos/mínimos y reacción del precio."
      : String(analysis.trend),

    liquidity_reading: badValue(analysis.liquidity_reading)
      ? "Liquidez evaluada en máximos/mínimos recientes, barridas, soportes, resistencias y reacción institucional."
      : String(analysis.liquidity_reading),

    entry_zone: badValue(analysis.entry_zone) ? fallbackEntry : String(analysis.entry_zone),

    stop_loss: badValue(analysis.stop_loss)
      ? stopPrice
        ? formatPrice(stopPrice, symbol, marketType)
        : "Stop técnico detrás de la zona de invalidación estructural más cercana."
      : String(analysis.stop_loss),

    take_profit_1: badValue(analysis.take_profit_1)
      ? tp1Price
        ? formatPrice(tp1Price, symbol, marketType)
        : "TP1 en la primera zona lógica de reacción a favor del movimiento."
      : String(analysis.take_profit_1),

    take_profit_2: badValue(analysis.take_profit_2)
      ? tp2Price
        ? formatPrice(tp2Price, symbol, marketType)
        : "TP2 en la siguiente zona de liquidez o extensión del movimiento."
      : String(analysis.take_profit_2),

    invalidation_zone: badValue(analysis.invalidation_zone)
      ? stopPrice
        ? `Invalidación si el precio rompe y sostiene más allá de ${formatPrice(
            stopPrice,
            symbol,
            marketType
          )}.`
        : "Invalidación si el precio rompe contra la estructura que justifica la operación."
      : String(analysis.invalidation_zone),

    confirmation: badValue(analysis.confirmation)
      ? signal === "BUY"
        ? "Confirmar BUY con rechazo alcista, ruptura válida, BOS/CHoCH o retesteo limpio antes de ejecutar."
        : signal === "SELL"
        ? "Confirmar SELL con rechazo bajista, ruptura válida, BOS/CHoCH o retesteo limpio antes de ejecutar."
        : "Esperar confirmación clara antes de operar."
      : String(analysis.confirmation),

    execution_trigger: badValue(analysis.execution_trigger)
      ? signal === "BUY"
        ? "Ejecutar BUY solo si aparece vela de intención alcista, rechazo fuerte o retesteo confirmado."
        : signal === "SELL"
        ? "Ejecutar SELL solo si aparece vela de intención bajista, rechazo fuerte o retesteo confirmado."
        : "No ejecutar hasta que el mercado defina dirección."
      : String(analysis.execution_trigger),

    missing_confirmation: badValue(analysis.missing_confirmation)
      ? "Falta una confirmación clara de desplazamiento, rechazo, ruptura o retesteo."
      : String(analysis.missing_confirmation),

    no_trade_condition: badValue(analysis.no_trade_condition)
      ? "No operar si el precio queda lateral, sin volumen, sin reacción clara o rompe la invalidación."
      : String(analysis.no_trade_condition),

    reading: badValue(analysis.reading)
      ? "Lectura técnica construida desde tendencia, estructura, liquidez, zonas de reacción y gestión de riesgo."
      : String(analysis.reading),

    institutional_summary: badValue(analysis.institutional_summary)
      ? "El setup debe tratarse como condicional. La entrada solo es válida si el precio confirma la dirección esperada."
      : String(analysis.institutional_summary),

    risk_note: badValue(analysis.risk_note)
      ? "Usar riesgo controlado. No perseguir el precio. Ejecutar solo con confirmación y stop definido."
      : String(analysis.risk_note),
  };
}

function buildInstitutionalPrompt({ marketType, symbol, mode, strategy }) {
  return `
Eres TradeMind AI, un motor institucional multi-mercado para análisis técnico profesional.

Mercado declarado: ${marketType}
Símbolo declarado: ${symbol}
Modo de análisis: ${mode}
Estrategia elegida por el usuario: ${strategy}

OBJETIVO:
Analizar el screenshot del gráfico y devolver un plan técnico accionable para traders de futuros, forex, crypto o acciones.

PERSONALIDAD OPERATIVA:
- Trader institucional.
- Agresivo pero lógico.
- No conservador en exceso.
- No genérico.
- No educativo.
- No inventes certeza absoluta.
- Construye escenarios operables cuando exista estructura razonable.

REGLA CRÍTICA SOBRE NEUTRAL:
- NEUTRAL solo está permitido si el gráfico está lateral, ilegible, sin dirección, sin estructura o sin zona técnica.
- Si existe tendencia, ruptura, retroceso, rechazo, liquidez, soporte/resistencia, BOS, CHoCH o zona de reacción, debes elegir BUY o SELL.
- Si la entrada aún no está lista, NO uses "No aplicable"; crea una entrada condicional profesional.

MERCADOS:
FUTURES:
- Prioriza MNQ/NQ/ES/MES, sesión de New York, premarket, overnight high/low, liquidez, rupturas falsas, retesteos, impulso y VWAP si aparece.

FOREX:
- Prioriza London/New York, liquidity sweep, BOS, CHoCH, order blocks, supply/demand, zonas psicológicas, spreads y confirmación.
- No uses lógica exclusiva de futuros en forex.

CRYPTO:
- Prioriza volatilidad, manipulación, barridas, soporte/resistencia, estructura y zonas de reacción.

REGLAS ABSOLUTAS:
1. Devuelve SOLO JSON válido.
2. No escribas texto fuera del JSON.
3. No dejes campos vacíos.
4. Prohibido usar: "No aplicable", "No hay entrada", "No se puede", "n/a", "null".
5. No abuses de NEUTRAL.
6. Si hay setup razonable, decide BUY o SELL.
7. Siempre entrega entry_zone, stop_loss, take_profit_1, take_profit_2, confirmation, invalidation_zone y no_trade_condition.
8. Si la entrada es condicional, dilo claramente.
9. El stop debe ir detrás de la invalidación técnica.
10. TP1 debe ser la primera zona lógica.
11. TP2 debe ser extensión o siguiente zona de liquidez.
12. ai_score debe reflejar calidad real del setup.
13. setup_quality NO TRADE solo si realmente no hay operación válida.

Devuelve exactamente este JSON:

{
  "market_type": "FUTURES | FOREX | CRYPTO | STOCKS | UNKNOWN",
  "symbol": "${symbol}",
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

app.get("/", (req, res) => {
  return res.json({
    ok: true,
    message: "TradeMind AI Backend Running",
    engine: ENGINE_VERSION,
    version: "5.0.0",
    model: OPENAI_MODEL,
  });
});

app.post("/analyze-chart", upload.any(), async (req, res) => {
  try {
    console.log("====================================");
    console.log("🔥 NUEVO ANÁLISIS TRADEMIND AI");
    console.log("ENGINE:", ENGINE_VERSION);
    console.log("MODEL:", OPENAI_MODEL);
    console.log("BODY:", req.body);
    console.log("FILES:", req.files?.map((f) => f.fieldname));
    console.log("====================================");

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        ok: false,
        error: "OPENAI_API_KEY no configurada.",
        analysis: forceAnalysis({}, req.body),
      });
    }

    const uploadedFile =
      req.files?.find((file) => file.fieldname === "chart") ||
      req.files?.find((file) => file.fieldname === "image") ||
      req.files?.[0];

    if (!uploadedFile) {
      return res.status(400).json({
        ok: false,
        error: "No se recibió imagen del gráfico.",
        analysis: forceAnalysis({}, req.body),
      });
    }

    const symbol = normalizeText(req.body?.symbol || req.body?.market || "UNKNOWN").toUpperCase();

    const marketType =
      normalizeMarketType(req.body?.marketType) !== "UNKNOWN"
        ? normalizeMarketType(req.body?.marketType)
        : inferMarketTypeFromSymbol(symbol);

    const mode = normalizeText(req.body?.mode || "Automático");
    const strategy = normalizeText(req.body?.strategy || "General");

    const base64Image = uploadedFile.buffer.toString("base64");

    const prompt = buildInstitutionalPrompt({
      marketType,
      symbol,
      mode,
      strategy,
    });

    const completion = await client.chat.completions.create({
      model: OPENAI_MODEL,
      temperature: 0.35,
      max_tokens: 1400,
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: prompt,
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Analiza este gráfico y devuelve únicamente JSON válido.",
            },
            {
              type: "image_url",
              image_url: {
                url: `data:${uploadedFile.mimetype || "image/jpeg"};base64,${base64Image}`,
              },
            },
          ],
        },
      ],
    });

    const rawText = completion.choices?.[0]?.message?.content || "";
    const parsed = safeJsonParse(rawText);

    const finalAnalysis = forceAnalysis(parsed, {
      marketType,
      symbol,
    });

    console.log("======== RAW OPENAI ========");
    console.log(rawText);
    console.log("======== FINAL ANALYSIS ========");
    console.log(finalAnalysis);

    return res.json({
      ok: true,
      engine: ENGINE_VERSION,
      version: "5.0.0",
      model: OPENAI_MODEL,
      symbol,
      marketType,
      mode,
      strategy,
      analysis: finalAnalysis,
    });
  } catch (error) {
    console.error("🔥 Error en /analyze-chart");
    console.error("MESSAGE:", error.message);
    console.error("DETAILS:", error.response?.data || error);

    return res.status(500).json({
      ok: false,
      engine: ENGINE_VERSION,
      error: error.message || "Error interno en análisis.",
      analysis: forceAnalysis({}, req.body),
    });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`${ENGINE_VERSION} corriendo en puerto ${PORT}`);
});
