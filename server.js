import dotenv from "dotenv";
dotenv.config();

import express from "express";
import multer from "multer";
import cors from "cors";
import OpenAI from "openai";

const app = express();

app.use(cors());
app.use(express.json({ limit: "30mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024,
  },
});

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const PORT = process.env.PORT || 3000;
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

const TWELVE_DATA_API_KEY = process.env.TWELVE_DATA_API_KEY;

const DATABENTO_API_KEY = process.env.DATABENTO_API_KEY;

const ENGINE_VERSION = "trademind-institutional-v3.1-premium";

const ANALYSIS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: [
    "market_type",
    "symbol",
    "signal",
    "confidence",
    "ai_score",
    "setup_quality",
    "risk_level",
    "market_context",
    "trend",
    "liquidity_reading",
    "entry_zone",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "invalidation_zone",
    "confirmation",
    "execution_trigger",
    "missing_confirmation",
    "no_trade_condition",
    "reading",
    "institutional_summary",
    "risk_note",
  ],
  properties: {
    market_type: {
      type: "string",
      enum: ["FUTURES", "FOREX", "CRYPTO", "STOCKS", "UNKNOWN"],
    },
    symbol: { type: "string" },
    signal: {
      type: "string",
      enum: ["BUY", "SELL", "NEUTRAL"],
    },
    confidence: {
      type: "string",
      enum: ["Alta", "Media", "Baja"],
    },
    ai_score: {
      type: "integer",
      minimum: 0,
      maximum: 100,
    },
    setup_quality: {
      type: "string",
      enum: ["A", "B", "C", "NO TRADE"],
    },
    risk_level: {
      type: "string",
      enum: ["Bajo", "Medio", "Alto"],
    },
    market_context: { type: "string" },
    trend: { type: "string" },
    liquidity_reading: { type: "string" },
    entry_zone: { type: "string" },
    stop_loss: { type: "string" },
    take_profit_1: { type: "string" },
    take_profit_2: { type: "string" },
    invalidation_zone: { type: "string" },
    confirmation: { type: "string" },
    execution_trigger: { type: "string" },
    missing_confirmation: { type: "string" },
    no_trade_condition: { type: "string" },
    reading: { type: "string" },
    institutional_summary: { type: "string" },
    risk_note: { type: "string" },
  },
};

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
    text.includes("no hay entrada") ||
    text.includes("no se puede") ||
    text.includes("sin definir") ||
    text.includes("no definido")
  );
}

function extractPrices(text) {
  if (!text) return [];

  const matches = String(text).match(/\d+(?:\.\d+)?/g);

  if (!matches) return [];

  return matches
    .map(Number)
    .filter((value) => Number.isFinite(value) && value > 0);
}

function average(prices) {
  if (!prices.length) return null;

  return prices.reduce((a, b) => a + b, 0) / prices.length;
}

function clampScore(value) {
  const score = Number(value);

  if (Number.isNaN(score)) return 58;

  return Math.max(0, Math.min(100, Math.round(score)));
}

function normalizeSignal(value) {
  const text = normalizeText(value).toUpperCase();

  if (
    text.includes("BUY") ||
    text.includes("COMPRA") ||
    text.includes("ALCISTA") ||
    text.includes("LONG")
  ) {
    return "BUY";
  }

  if (
    text.includes("SELL") ||
    text.includes("VENTA") ||
    text.includes("BAJISTA") ||
    text.includes("SHORT")
  ) {
    return "SELL";
  }

  return "NEUTRAL";
}

function normalizeMarketType(value) {
  const text = normalizeText(value).toUpperCase();

  if (text.includes("FUTURE")) return "FUTURES";
  if (text.includes("FOREX") || text.includes("FX")) return "FOREX";
  if (text.includes("CRYPTO")) return "CRYPTO";
  if (text.includes("STOCK")) return "STOCKS";

  return "UNKNOWN";
}

function inferMarketTypeFromSymbol(symbol = "") {
  const s = normalizeText(symbol).toUpperCase().replace("/", "");

  const futures = ["MNQ", "NQ", "ES", "MES", "YM", "MYM", "RTY", "M2K"];
  const forex = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
    "EURJPY",
    "GBPJPY",
    "XAUUSD",
    "XAGUSD",
  ];
  const crypto = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "BTCUSDT", "ETHUSDT"];

  if (futures.includes(s)) return "FUTURES";
  if (forex.includes(s)) return "FOREX";
  if (crypto.includes(s)) return "CRYPTO";

  return "UNKNOWN";
}

function getPriceDecimals(symbol = "", marketType = "", price = null) {
  const s = normalizeText(symbol).toUpperCase().replace("/", "");
  const type = normalizeMarketType(marketType);

  if (s.includes("JPY")) return 3;
  if (s === "XAUUSD" || s === "XAGUSD") return 2;
  if (type === "FOREX") return 5;
  if (type === "CRYPTO") return 2;
  if (type === "FUTURES") return 2;
  if (price && price > 0 && price < 100) return 5;

  return 2;
}

function formatPrice(price, symbol = "", marketType = "") {
  if (price === null || price === undefined || Number.isNaN(Number(price))) {
    return null;
  }

  return Number(price).toFixed(getPriceDecimals(symbol, marketType, price));
}

function normalizePriceField(value, symbol = "", marketType = "") {
  if (badValue(value)) return value;

  const prices = extractPrices(value);

  if (!prices.length) return value;

  const formattedPrices = prices
    .map((price) => formatPrice(price, symbol, marketType))
    .filter(Boolean);

  if (!formattedPrices.length) return value;

  if (formattedPrices.length === 1) return formattedPrices[0];

  return formattedPrices.join(" - ");
}

function getDefaultBuffer(entryPrice, symbol = "", marketType = "") {
  const s = normalizeText(symbol).toUpperCase().replace("/", "");
  const type = normalizeMarketType(marketType);

  if (s === "XAUUSD") return 2.5;
  if (s.includes("JPY")) return 0.15;
  if (type === "FOREX") return 0.0012;
  if (type === "CRYPTO") return entryPrice ? entryPrice * 0.006 : 50;
  if (type === "FUTURES") return 25;

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

  if (aiScore >= 55) return "Medio";

  return "Alto";
}

function forceDirectionalSignal(analysis = {}) {
  const rawSignal = normalizeSignal(analysis.signal);

  if (rawSignal !== "NEUTRAL") return rawSignal;

  const combined = [
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
    "long",
    "soporte",
    "retesteó soporte",
    "rechazo alcista",
    "ruptura alcista",
    "bos alcista",
    "choch alcista",
    "higher low",
    "demanda",
    "demand",
  ];

  const bearishWords = [
    "bajista",
    "sell",
    "venta",
    "short",
    "resistencia",
    "rechazo bajista",
    "ruptura bajista",
    "bos bajista",
    "choch bajista",
    "lower high",
    "oferta",
    "supply",
  ];

  const bullishScore = bullishWords.filter((w) => combined.includes(w)).length;
  const bearishScore = bearishWords.filter((w) => combined.includes(w)).length;

  if (bullishScore >= bearishScore + 1) return "BUY";
  if (bearishScore >= bullishScore + 1) return "SELL";

  return "NEUTRAL";
}

function forceAnalysis(input = {}, metadata = {}) {
  const analysis = input && typeof input === "object" ? input : {};

  const symbol = normalizeText(
    analysis.symbol || metadata.symbol || "UNKNOWN"
  ).toUpperCase();

  const requestedMarketType = normalizeMarketType(metadata.marketType);
  const modelMarketType = normalizeMarketType(analysis.market_type);
  const inferredMarketType = inferMarketTypeFromSymbol(symbol);

  const marketType =
    requestedMarketType !== "UNKNOWN"
      ? requestedMarketType
      : modelMarketType !== "UNKNOWN"
      ? modelMarketType
      : inferredMarketType;

  let aiScore = clampScore(analysis.ai_score);
  let signal = forceDirectionalSignal(analysis);

  const entryPrices = extractPrices(analysis.entry_zone);
  const stopPrices = extractPrices(analysis.stop_loss);
  const tp1Prices = extractPrices(analysis.take_profit_1);
  const tp2Prices = extractPrices(analysis.take_profit_2);
  const invalidationPrices = extractPrices(analysis.invalidation_zone);

  const entryPrice = average(entryPrices);

  let stopPrice = stopPrices.length ? stopPrices[0] : null;
  let tp1Price = tp1Prices.length ? tp1Prices[0] : null;
  let tp2Price = tp2Prices.length ? tp2Prices[0] : null;
  const invalidationPrice = invalidationPrices.length
    ? invalidationPrices[0]
    : null;

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
      ? `BUY condicional en ${marketLabel}: buscar entrada en retroceso hacia zona de demanda/soporte o tras ruptura y retesteo confirmado.`
      : signal === "SELL"
      ? `SELL condicional en ${marketLabel}: buscar entrada en retroceso hacia zona de oferta/resistencia o tras ruptura bajista y retesteo confirmado.`
      : `NO TRADE en ${marketLabel}: esperar ruptura clara, rechazo institucional o dirección definida antes de operar.`;

  if (signal !== "NEUTRAL" && aiScore < 58) {
    aiScore = 58;
  }

  return {
    market_type: marketType,
    symbol: symbol || "UNKNOWN",
    signal,
    confidence: normalizeConfidence(analysis.confidence, aiScore),
    ai_score: aiScore,
    setup_quality: normalizeSetupQuality(
      analysis.setup_quality,
      aiScore,
      signal
    ),
    risk_level: normalizeRiskLevel(analysis.risk_level, aiScore),

    market_context: badValue(analysis.market_context)
      ? `Contexto ${marketLabel}: lectura basada en estructura visible, zonas de reacción, liquidez y dirección probable.`
      : analysis.market_context,

    trend: badValue(analysis.trend)
      ? "Tendencia evaluada por estructura reciente, máximos/mínimos y reacción del precio."
      : analysis.trend,

    liquidity_reading: badValue(analysis.liquidity_reading)
      ? "Liquidez evaluada en máximos/mínimos recientes, zonas de barrida, soportes, resistencias y reacción institucional."
      : analysis.liquidity_reading,

    entry_zone: badValue(analysis.entry_zone)
      ? fallbackEntry
      : normalizePriceField(analysis.entry_zone, symbol, marketType),

    stop_loss: badValue(analysis.stop_loss)
      ? stopPrice
        ? formatPrice(stopPrice, symbol, marketType)
        : "Stop técnico detrás de la zona de invalidación estructural."
      : normalizePriceField(analysis.stop_loss, symbol, marketType),

    take_profit_1: badValue(analysis.take_profit_1)
      ? tp1Price
        ? formatPrice(tp1Price, symbol, marketType)
        : "TP1 en la primera zona lógica de reacción a favor del movimiento."
      : normalizePriceField(analysis.take_profit_1, symbol, marketType),

    take_profit_2: badValue(analysis.take_profit_2)
      ? tp2Price
        ? formatPrice(tp2Price, symbol, marketType)
        : "TP2 en la siguiente zona de liquidez o extensión del movimiento."
      : normalizePriceField(analysis.take_profit_2, symbol, marketType),

    invalidation_zone: badValue(analysis.invalidation_zone)
      ? stopPrice
        ? `Invalidación si el precio rompe y sostiene más allá de ${formatPrice(
            stopPrice,
            symbol,
            marketType
          )}.`
        : "Invalidación si el precio rompe contra la estructura que justifica la operación."
      : normalizePriceField(analysis.invalidation_zone, symbol, marketType),

    confirmation: badValue(analysis.confirmation)
      ? signal === "BUY"
        ? "Confirmar BUY con rechazo alcista, ruptura válida, BOS/CHoCH o retesteo limpio antes de ejecutar."
        : signal === "SELL"
        ? "Confirmar SELL con rechazo bajista, ruptura válida, BOS/CHoCH o retesteo limpio antes de ejecutar."
        : "Esperar confirmación clara antes de operar."
      : analysis.confirmation,

    execution_trigger: badValue(analysis.execution_trigger)
      ? signal === "BUY"
        ? "Ejecutar BUY solo si aparece vela de intención alcista, rechazo fuerte o retesteo confirmado."
        : signal === "SELL"
        ? "Ejecutar SELL solo si aparece vela de intención bajista, rechazo fuerte o retesteo confirmado."
        : "No ejecutar hasta que el mercado defina dirección."
      : analysis.execution_trigger,

    missing_confirmation: badValue(analysis.missing_confirmation)
      ? "Falta una confirmación clara de desplazamiento, rechazo, ruptura o retesteo."
      : analysis.missing_confirmation,

    no_trade_condition: badValue(analysis.no_trade_condition)
      ? "No operar si el precio queda lateral, sin volumen, sin reacción clara o rompe la invalidación."
      : analysis.no_trade_condition,

    reading: badValue(analysis.reading)
      ? "Lectura técnica construida desde tendencia, estructura, liquidez, zonas de reacción y gestión de riesgo."
      : analysis.reading,

    institutional_summary: badValue(analysis.institutional_summary)
      ? "El setup debe tratarse como condicional. La entrada solo es válida si el precio confirma la dirección esperada."
      : analysis.institutional_summary,

    risk_note: badValue(analysis.risk_note)
      ? "Usar riesgo controlado. No perseguir el precio. Ejecutar solo con confirmación y stop definido."
      : analysis.risk_note,
  };
}
function buildInstitutionalPrompt({
  marketType = "UNKNOWN",
  symbol = "UNKNOWN",
  marketContext = null,
} = {}) {

  const cleanMarketType = normalizeMarketType(marketType);
  const cleanSymbol = normalizeText(symbol || "UNKNOWN").toUpperCase();
const formattedMarketContext =
  Array.isArray(marketContext) && marketContext.length > 0
    ? marketContext
        .slice(0, 100)
        .map((candle) => ({
          datetime: candle.datetime,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }))
    : null;

const marketContextText = formattedMarketContext
  ? JSON.stringify(formattedMarketContext)
  : "DATOS DE MERCADO NO DISPONIBLES";
  return `
Eres TradeMind AI, un motor institucional multi-mercado para análisis técnico profesional.

Mercado declarado: ${cleanMarketType}
Símbolo declarado: ${cleanSymbol}
DATOS DE MERCADO EN TIEMPO REAL:

${Array.isArray(marketContext)
  ? JSON.stringify(
      marketContext.slice(0, 10),
      null,
      2
    )
  : "No disponibles"}
REGLAS PARA USAR LOS DATOS REALES:
- Los datos están ordenados desde la vela más reciente hacia las anteriores.
- Usa estos datos para confirmar tendencia, máximos, mínimos, impulso, retrocesos y volatilidad.
- Cruza el contexto histórico con la imagen enviada.
- Si existe contradicción entre la imagen y los datos reales, reduce la confianza.
- No inventes precios fuera del rango razonable observado.
- Prioriza siempre la estructura visible del gráfico y utiliza los datos históricos como confirmación.
OBJETIVO:
Analizar el screenshot del gráfico y devolver un plan técnico accionable para traders de futuros, forex, crypto o acciones.

PERSONALIDAD OPERATIVA:
- Trader institucional.
- Agresivo, pero lógico.
- No conservador en exceso.
- No genérico.
- No educativo.
- No inventes certeza absoluta.
- Construye escenarios operables cuando exista estructura razonable.

REGLA CRÍTICA DE PRECISIÓN DE PRECIOS:
- Si el mercado es FOREX, devuelve precios con 5 decimales exactos. Ejemplo correcto: 1.15771. Ejemplo incorrecto: 1.1577.
- Si el par contiene JPY, devuelve 3 decimales. Ejemplo correcto: 156.245.
- Si es XAUUSD o XAGUSD, devuelve 2 decimales.
- Si es FUTURES como MNQ, NQ, ES o MES, devuelve precios con 2 decimales.
- Nunca redondees visualmente los precios.
- Lee los precios del eje derecho del gráfico y usa niveles visibles reales.
- No inventes números redondos como 1.1550 o 1.1530 si el gráfico muestra niveles más precisos.
- La entrada, stop loss, take profit 1 y take profit 2 deben ser precios numéricos concretos siempre que el gráfico tenga una escala visible.
- Si el precio actual se ve en el gráfico, úsalo como referencia para que la entrada tenga sentido con la zona actual del mercado.
REGLA CRÍTICA DE CALIDAD OPERATIVA:
- No propongas una entrada simplemente por estar cerca del precio actual.
- La entrada debe estar basada en una zona técnica real: resistencia, soporte, order block, supply/demand, ruptura + retesteo, sweep de liquidez o rechazo visible.
- Si la entrada está demasiado lejos o el precio ya se movió demasiado, marca la entrada como condicional.
- Para SELL:
  - La entrada debe estar cerca de resistencia, retroceso, retesteo bajista o rechazo.
  - El stop debe ir por encima del máximo/rechazo/zona de invalidación.
  - TP1 debe buscar mínimo la primera zona lógica de liquidez.
  - TP2 debe buscar una extensión razonable o siguiente soporte.
- Para BUY:
  - La entrada debe estar cerca de soporte, retroceso, retesteo alcista o rechazo.
  - El stop debe ir por debajo del mínimo/rechazo/zona de invalidación.
  - TP1 debe buscar mínimo la primera zona lógica de liquidez.
  - TP2 debe buscar una extensión razonable o siguiente resistencia.
- No des operaciones con TP demasiado pequeño si el riesgo no justifica la entrada.
- Antes de devolver BUY o SELL, valida que exista una relación riesgo/beneficio lógica.
- Si la operación no ofrece al menos una estructura razonable cercana a 1:1 hacia TP1, baja la confianza o marca setup_quality como C.
- Si el precio está en medio del rango sin ventaja clara, usa NEUTRAL o entrada condicional.
REGLA CRÍTICA SOBRE NEUTRAL:
- NEUTRAL solo está permitido si el gráfico está lateral, ilegible, sin dirección, sin estructura o sin zona técnica.
- Si existe tendencia, ruptura, retroceso, rechazo, liquidez, soporte/resistencia, BOS, CHoCH o zona de reacción, debes elegir BUY o SELL.
- Si la entrada aún no está lista, NO uses "No aplicable"; crea una entrada condicional profesional.

MERCADOS:
FUTURES:
- Prioriza MNQ/NQ/ES/MES, sesión de New York, premarket, overnight high/low, liquidez, rupturas falsas, retesteos, impulso, VWAP si aparece.

FOREX:
- Prioriza London/New York, liquidity sweep, BOS, CHoCH, order blocks, supply/demand, zonas psicológicas, spreads y confirmación.
- No uses lógica exclusiva de futuros en forex.
- Para EURUSD, GBPUSD, AUDUSD, NZDUSD y pares similares, usa 5 decimales.
- Para pares JPY usa 3 decimales.

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
14. entry_zone, stop_loss, take_profit_1 y take_profit_2 deben respetar la cantidad de decimales del mercado.
15. No devuelvas precios incompletos.,
16. La entrada no debe ser un número inventado; debe tener sentido con la estructura visible.
17. El stop loss debe invalidar la idea, no solo estar cerca del precio.
18. TP1 y TP2 deben estar a favor del movimiento y en zonas lógicas.
19. No fuerces operaciones si el precio está en una zona de mala entrada.
20. Si el precio ya llegó tarde al movimiento, explica que se debe esperar retroceso o nueva confirmación.

Responde únicamente con el JSON solicitado por el sistema.
`;
}

function getFallbackResponse(req, message = "No se pudo completar el análisis.") {
  return {
    ok: false,
    engine: ENGINE_VERSION,
    error: message,
    analysis: forceAnalysis(
      {},
      {
        marketType: req.body?.marketType,
        symbol: req.body?.symbol,
      }
    ),
  };
}

app.get("/", (req, res) => {
  res.json({
    ok: true,
    message: "TradeMind AI Backend Running",
    engine: ENGINE_VERSION,
    version: "3.1.0",
    model: OPENAI_MODEL,
    markets: ["FUTURES", "FOREX", "CRYPTO", "STOCKS"],
  });
});
async function getForexData(symbol) {
  try {
    const cleanSymbol = symbol
      .toString()
      .trim()
      .toUpperCase()
      .replaceAll("/", "")
      .replaceAll(" ", "");

    const formattedSymbol =
      cleanSymbol.length === 6
        ? `${cleanSymbol.substring(0, 3)}/${cleanSymbol.substring(3, 6)}`
        : cleanSymbol;

    const response = await fetch(
      `https://api.twelvedata.com/time_series` +
        `?symbol=${encodeURIComponent(formattedSymbol)}` +
        `&interval=1h` +
        `&outputsize=100` +
        `&apikey=${TWELVE_DATA_API_KEY}`
    );

    const data = await response.json();

    if (!response.ok || !Array.isArray(data.values)) {
      throw new Error(
        data?.message ||
          `No se pudieron obtener datos para ${formattedSymbol}`
      );
    }

    console.log("📈 DATOS DE TWELVE DATA:");
    console.log("Símbolo recibido:", symbol);
    console.log("Símbolo enviado:", formattedSymbol);
    console.log("Velas recibidas:", data.values.length);
    console.log("Última vela:", data.values[0]);

    return data.values;
  } catch (error) {
    console.error(
      "Error obteniendo datos de Twelve Data:",
      error
    );

    return null;
  }
}
function mapDatabentoFutureSymbol(symbol) {
  const cleanSymbol = String(symbol || "")
    .trim()
    .toUpperCase()
    .replaceAll("/", "")
    .replaceAll(" ", "");

  const supportedSymbols = {
    MNQ: "MNQ.v.0",
    NQ: "NQ.v.0",
    MES: "MES.v.0",
    ES: "ES.v.0",
    MYM: "MYM.v.0",
    YM: "YM.v.0",
    M2K: "M2K.v.0",
    RTY: "RTY.v.0",
    MGC: "MGC.v.0",
    GC: "GC.v.0",
    MCL: "MCL.v.0",
    CL: "CL.v.0",
  };

  return supportedSymbols[cleanSymbol] || null;
}
async function getFuturesData(symbol) {
  try {
    const databentoSymbol =
      mapDatabentoFutureSymbol(symbol);

    if (!databentoSymbol) {
      console.log(
        "⚠️ Símbolo no soportado:",
        symbol
      );

      return null;
    }

    console.log("📈 DATOS DE DATABENTO:");
    console.log("Símbolo recibido:", symbol);
    console.log(
      "Símbolo convertido:",
      databentoSymbol
    );

    const response = await fetch(
      "https://hist.databento.com/v0/timeseries.get_range",
      {
        method: "POST",

        headers: {
          Authorization:
            "Basic " +
            Buffer.from(
              `${DATABENTO_API_KEY}:`
            ).toString("base64"),

          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          dataset: "GLBX.MDP3",

          schema: "ohlcv-1h",

          stype_in: "continuous",

          symbols: [databentoSymbol],

          limit: 100,
        }),
      }
    );

    const data = await response.json();

    console.log(
      "Respuesta Databento:"
    );

    console.log(data);

    return data;
  } catch (error) {
    console.error(
      "❌ Error Databento:",
      error
    );

    return null;
  }
}
app.post("/analyze-chart", upload.single("chart"), async (req, res) => {
  console.log("🚨 SOLICITUD RECIBIDA EN /analyze-chart");
  console.log("BODY:", req.body);
  console.log("ARCHIVO RECIBIDO:", Boolean(req.file));
  console.log("FECHA:", new Date().toISOString());

  try {

    if (!req.file) {
      return res
        .status(400)
        .json(getFallbackResponse(req, "No se recibió imagen."));
    }

    const marketType =
      req.body?.marketType || req.body?.market_type || "UNKNOWN";

    const symbol = req.body?.symbol || "UNKNOWN";

    const normalizedMarketType =
      normalizeMarketType(marketType) !== "UNKNOWN"
        ? normalizeMarketType(marketType)
        : inferMarketTypeFromSymbol(symbol);
let marketContext = null;

if (normalizedMarketType === "FOREX") {
  marketContext = await getForexData(symbol);
}

if (normalizedMarketType === "FUTURES") {
  marketContext = await getFuturesData(symbol);
}

if (normalizedMarketType === "CRYPTO") {
  // Próximamente conectaremos el proveedor cripto.
  marketContext = null;
}
    const base64Image = req.file.buffer.toString("base64");
const prompt = buildInstitutionalPrompt({
  marketType: normalizedMarketType,
  symbol,
  marketContext,
});
   

    console.log("====================================");
    console.log("🔥 NUEVO ANÁLISIS TRADEMIND AI");
    console.log("ENGINE:", ENGINE_VERSION);
    console.log("MODEL:", OPENAI_MODEL);
    console.log("MARKET:", normalizedMarketType);
    console.log("SYMBOL:", symbol);
    console.log("FILE SIZE:", req.file.size);
    console.log("====================================");

    const response = await client.responses.create({
      model: OPENAI_MODEL,
      max_output_tokens: 1300,
      temperature: 0.28,
      text: {
        format: {
          type: "json_schema",
          name: "trademind_chart_analysis",
          strict: true,
          schema: ANALYSIS_SCHEMA,
        },
      },
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: prompt,
            },
            {
              type: "input_image",
              image_url: `data:${
                req.file.mimetype || "image/jpeg"
              };base64,${base64Image}`,
            },
          ],
        },
      ],
    });

    const rawText = response.output_text || "";
    const parsedAnalysis = safeJsonParse(rawText);

    console.log("======== RAW OPENAI ========");
    console.log(rawText);

    console.log("======== PARSED ========");
    console.log(parsedAnalysis);

    const finalAnalysis = forceAnalysis(parsedAnalysis, {
      marketType: normalizedMarketType,
      symbol,
    });

    console.log("======== FINAL ANALYSIS ========");
    console.log(finalAnalysis);

    return res.json({
      ok: true,
      engine: ENGINE_VERSION,
      model: OPENAI_MODEL,
      analysis: finalAnalysis,
    });
  } catch (error) {
    console.error("🔥 Error en /analyze-chart");
    console.error("STATUS:", error.status);
    console.error("MESSAGE:", error.message);
    console.error("DETAILS:", error.response?.data || error);

    return res
      .status(500)
      .json(getFallbackResponse(req, "Error interno en análisis."));
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🔥 ${ENGINE_VERSION} corriendo en puerto ${PORT}`);
});
