const express = require('express');
const cors = require('cors');
const multer = require('multer');
const OpenAI = require('openai');
require('dotenv').config();

const app = express();

app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.get('/', (req, res) => {
  res.send('TradeMind AI Backend Online');
});

app.post('/analyze-chart', upload.single('image'), async (req, res) => {
  try {
    const base64Image = req.file.buffer.toString('base64');

    const response = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `
Eres un trader profesional institucional especializado en Nasdaq futures.

Tu trabajo es analizar gráficos y devolver SIEMPRE un análisis COMPLETO y PROFESIONAL en formato JSON.

OBLIGATORIO:
- trend
- setup
- entry
- stopLoss
- takeProfit1
- takeProfit2
- confirmation
- riskLevel
- explanation
- probability
- noTradeCondition

Nunca dejes campos vacíos.
Nunca respondas fuera del JSON.
`,
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'Analiza este gráfico.',
            },
            {
              type: 'image_url',
              image_url: {
                url: `data:image/png;base64,${base64Image}`,
              },
            },
          ],
        },
      ],
      max_tokens: 1000,
    });

    const result = response.choices[0].message.content;

    res.json({
      success: true,
      analysis: result,
    });
  } catch (error) {
    console.log(error);

    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`Servidor corriendo en puerto ${PORT}`);
});
