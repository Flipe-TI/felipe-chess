# felipe-chess

**Um motor de xadrez que joga como eu.** Uma rede neural de política treinada por
*behavior cloning* das minhas partidas reais no Chess.com — treinada para prever
**o lance que eu jogaria**, não o melhor lance. O modelo roda no navegador (ONNX)
na demo jogável do meu portfólio.

▶️ **Jogue contra o bot:** [curriculo.felipegabriel.dev](https://curriculo.felipegabriel.dev) → seção *"Joga comigo"*

> Não é um motor forte — é isso que é o ponto. Ele imita meu estilo, minhas
> aberturas e meu nível (~850 rapid). Fraco e autêntico > forte e genérico.

---

## Por que este projeto

É um case de **Data Ops / ML end-to-end**, autocontido e reprodutível:

**dados públicos → dataset → treino → deploy no navegador → MLOps**

Cada elo é um script pequeno e testado. O contrato entre o Python (treino) e o
JavaScript (inferência no site) é blindado por *parity tests* — porque o risco
central de um projeto assim é a **divergência silenciosa**: se o encoding do
Python e o do JS discordarem em 1 bit, o modelo joga lixo sem erro nenhum.

## Estrutura do projeto

```
felipe-chess/
├── src/felipe_chess/
│   ├── encoding.py         # posição→tensor 18×8×8 / lance→índice (espelha o encoding.mjs do site)
│   ├── fetch_chesscom.py   # baixa minhas partidas da API pública do Chess.com (idempotente)
│   ├── build_dataset.py    # PGN → amostras (posição, lance) — só os MEUS lances
│   ├── model.py            # PolicyNet: CNN residual pequena (policy head)
│   ├── train.py            # treino + early stopping + move-match no holdout
│   └── export_onnx.py      # export ONNX + parity numérica ONNX↔PyTorch
├── tests/                  # parity (JS↔Python, ONNX↔PyTorch) + unit tests
│   └── fixtures/           # parity.json (do site) + js_oracle.json (gerado do encoding.mjs)
├── tools/gen_js_oracle.mjs # gera o oráculo branch-complete a partir do JS do site
├── data/felipe/            # minhas partidas (PGN, versionado)
└── requirements.txt · pyproject.toml
```

## Pipeline

```
 Chess.com API ──▶ fetch_chesscom ──▶ data/felipe/*.pgn   (partidas versionadas)
                                          │
                                          ▼
                    build_dataset ──▶ (posição 18×8×8, índice do lance)   [só os MEUS lances]
                                          │
                                          ▼
        model.py (PolicyNet)  ──▶  train.py  ──▶  models/policy.pt   (early stopping)
                                          │
                                          ▼
                 export_onnx ──▶ model.onnx  ──(parity ONNX↔PyTorch)──▶  site
```

## O contrato (congelado)

O modelo fala exatamente a língua que o site espera:

| | |
|---|---|
| Input  | `board`  float32 `[1, 18, 8, 8]` |
| Output | `policy` float32 `[1, 4672]` (logits) |
| Encoding | AlphaZero 8×8×73, `encoding_version = az-8x8x73-v1` |
| Perspectiva | sempre do lado a mover (espelhamento para as pretas) |

**Blindagem contra divergência silenciosa:**
- `encoding.py` reproduz bit a bit o `encoding.mjs` do site. Verificado por dois
  níveis de parity test: a fixture oficial (`parity.json`, copiada do site) e um
  **oráculo branch-complete** gerado do próprio JS por `tools/gen_js_oracle.mjs`
  (`js_oracle.json`, 2.254 lances cobrindo todos os branches — queen / knight /
  underpromotion / promoção). Regenerar: `node tools/gen_js_oracle.mjs`.
- `export_onnx.py` valida o ONNX numericamente contra o PyTorch antes de
  publicar (max_diff ≈ 3e-05).

## Decisões de design

- **Behavior cloning, não força.** O objetivo é *parecer eu*, então o alvo do
  treino é o lance que eu joguei — e só nas posições onde é a minha vez. Os
  lances dos oponentes são descartados (senão eu diluiria meu estilo no deles).
- **Rede pequena de propósito.** `PolicyNet(64 filtros, 5 blocos)` = 385k params
  / 1.54 MB. Precisa caber num download de navegador, então YAGNI: só policy
  head, sem value head.
- **Só as minhas partidas, por ora.** A arquitetura prevê um pré-treino opcional
  num corpus público de baixo Elo (Lichess) para robustez, mas a primeira versão
  foi treinada **só** com os meus jogos, deliberadamente, para medir o sinal puro
  de estilo antes de decidir por essa camada.
- **Regularização > dados.** Com ~12k amostras o risco é overfit; a defesa é rede
  pequena + weight-decay + early stopping medido no holdout.
- **Contrato imutável.** O I/O `board→policy` é congelado; retreinos só trocam os
  pesos, nunca o formato — o site nunca quebra.

## Resultados (primeiro corte, só com as minhas partidas)

| Métrica | Valor |
|---|---|
| Dataset | 386 partidas rapid → **12.076 amostras** (só os meus lances) |
| Modelo | `PolicyNet` (5 blocos residuais, 64 filtros) — **385k params, 1.54 MB** |
| **Holdout top-1 move-match** | **31,5%** |
| **Holdout top-3 move-match** | **44,5%** |

*Numa posição há ~30 lances legais; acertar o meu lance exato 31,5% das vezes
está bem acima do acaso (~3%). A métrica é "parece comigo", não Elo.*

## Limitações

- **Poucos dados** de um só jogador → propenso a overfit (mitigado como acima).
- **Fraco por design** — reproduz os erros e hábitos de um jogador ~850. Feature,
  não bug.
- A saída são **logits** (sem softmax); o decode para lance concreto exige
  máscara de legalidade downstream (o site faz isso).
- Evolui com o tempo: conforme jogo mais, o dataset cresce e o modelo é
  retreinado.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt
# torch CPU (wheel +cpu):
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

pytest                                 # suíte completa (inclui os parity tests)

# pipeline (src-layout: rode com PYTHONPATH=src)
python -m felipe_chess.fetch_chesscom  --out data/felipe
python -m felipe_chess.build_dataset   --pgn-dir data/felipe --out data/processed
python -m felipe_chess.train           --data data/processed --out models
python -m felipe_chess.export_onnx     --checkpoint models/policy.pt --out models/model.onnx
```

## Stack

Python · PyTorch · python-chess · NumPy · ONNX / ONNX Runtime — e
`onnxruntime-web` do lado do site para a inferência no navegador.

## Dados & privacidade

Treinado exclusivamente com partidas **públicas do próprio autor** no Chess.com.

## Licença

MIT — ver [LICENSE](LICENSE).
