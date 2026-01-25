import { useEffect, useRef } from 'react'

interface GameState {
  board: number[][]
  current_player: number
  move_count: number
  last_move: [number, number] | null
  captured_black: number
  captured_white: number
  game_over: boolean
}

interface MinimalGoBoardProps {
  gameState: GameState
}