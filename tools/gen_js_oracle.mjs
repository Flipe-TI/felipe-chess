/**
 * Gera um oráculo branch-complete a partir do encoding.mjs do repo do site.
 * Escreve tests/fixtures/js_oracle.json. O parity test (Python) exige que o
 * encoding.py reproduza cada entrada — verificação cross-language de verdade,
 * cobrindo os branches que parity.json não cobre (promoções, todos os
 * from-squares, ambas as perspectivas).
 *
 * Uso: node tools/gen_js_oracle.mjs
 * Requer o repo irmão ao lado: ../Flipe-TI.github.io
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const sitePath = pathToFileURL(
  resolve(here, "../../Flipe-TI.github.io/assets/chess/encoding.mjs")
).href;
const {
  ENCODING_VERSION,
  POLICY_SIZE,
  encodePosition,
  moveToIndex,
  indexToMove,
  toPerspectiveSquare,
} = await import(sitePath);

const onBoard = (sq) => {
  const f = sq.charCodeAt(0) - 97;
  const r = parseInt(sq[1], 10) - 1;
  return f >= 0 && f < 8 && r >= 0 && r < 8;
};

// --- moves: todos os índices válidos e re-codificáveis ---------------------
// Para cada índice i, reconstrói o lance via indexToMove (JS) e confirma que
// moveToIndex (JS) devolve i. Guarda só os que ficam no tabuleiro e são
// estáveis no round-trip. Cobre todos os 64×73 branches.
const moves = [];
for (let i = 0; i < POLICY_SIZE; i++) {
  const m = indexToMove(i, "w");
  if (!onBoard(m.from) || !onBoard(m.to)) continue;
  let j;
  try {
    j = moveToIndex(m);
  } catch {
    continue;
  }
  if (j !== i) continue; // índice não é estável no round-trip; pula
  moves.push({ from: m.from, to: m.to, promotion: m.promotion, index: i });
}

// --- perspective: 64 casas × ambos os lados --------------------------------
const perspective = [];
for (let r = 1; r <= 8; r++) {
  for (let f = 0; f < 8; f++) {
    const sq = String.fromCharCode(97 + f) + r;
    for (const side of ["w", "b"]) {
      perspective.push({ square: sq, side, result: toPerspectiveSquare(sq, side) });
    }
  }
}

// --- positions: FENs diversos (EP e roque em ambas as perspectivas) --------
const fens = [
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
  "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3", // EP, pretas a mover
  "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3", // EP, brancas a mover
  "r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1",     // só roque branco rei
  "r3k2r/8/8/8/8/8/8/R3K2R b q - 0 1",     // só roque preto dama (perspectiva preta)
  "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  // todos os roques
  "8/8/8/4k3/4K3/8/8/8 w - - 0 1",         // sem roque
  "8/P6p/8/8/8/8/p6P/8 w - - 0 1",         // peças em rank de promoção, brancas
  "8/P6p/8/8/8/8/p6P/8 b - - 0 1",         // idem, pretas a mover (color-swap+flip)
  "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
  "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 6 5",
];
const positions = fens.map((fen) => {
  const t = encodePosition(fen);
  const nz = [];
  for (let i = 0; i < t.length; i++) if (t[i] !== 0) nz.push(i);
  return { fen, nonzero_indices: nz };
});

const oracle = { encoding_version: ENCODING_VERSION, moves, perspective, positions };
const outPath = resolve(here, "../tests/fixtures/js_oracle.json");
writeFileSync(outPath, JSON.stringify(oracle, null, 1) + "\n");
console.log(
  `js_oracle.json escrito: ${moves.length} moves, ${perspective.length} perspective, ${positions.length} positions`
);
