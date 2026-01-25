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

const MinimalGoBoard = ({ gameState }: MinimalGoBoardProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const cellSize = 40

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // calculate board size from board array
    const boardSize = gameState.board[0].length

    // set canvas size - include padding for edge stones
    const gridLines = boardSize + 2
    const padding = cellSize / 2
    const canvasSize = cellSize * (gridLines - 1) + cellSize
    canvas.width = canvasSize
    canvas.height = canvasSize

    // clear canvas with white background
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, canvasSize, canvasSize)

    // draw grid lines - stones go on intersections
    ctx.strokeStyle = '#808080'
    ctx.lineWidth = 1

    // draw vertical lines (skip outermost edges, fade at ends)
    for (let i = 1; i < gridLines - 1; i++) {
      const x = padding + i * cellSize
      const gradient = ctx.createLinearGradient(0, 0, 0, canvasSize)
      gradient.addColorStop(0, 'rgba(128, 128, 128, 0)')
      gradient.addColorStop(0.2, 'rgba(128, 128, 128, 0.8)')
      gradient.addColorStop(0.8, 'rgba(128, 128, 128, 0.8)')
      gradient.addColorStop(1, 'rgba(128, 128, 128, 0)')
      ctx.strokeStyle = gradient
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, canvasSize - padding)
      ctx.stroke()
    }

    // draw horizontal lines (skip outermost edges, fade at ends)
    for (let i = 1; i < gridLines - 1; i++) {
      const y = padding + i * cellSize
      const gradient = ctx.createLinearGradient(0, 0, canvasSize, 0)
      gradient.addColorStop(0, 'rgba(128, 128, 128, 0)')
      gradient.addColorStop(0.2, 'rgba(128, 128, 128, 0.8)')
      gradient.addColorStop(0.8, 'rgba(128, 128, 128, 0.8)')
      gradient.addColorStop(1, 'rgba(128, 128, 128, 0)')
      ctx.strokeStyle = gradient
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(canvasSize - padding, y)
      ctx.stroke()
    }

    // star points (hoshi)
    const getStarPoints = (size: number): [number, number][] => {
      if (size === 9) return [[3, 3], [7, 3], [5, 5], [3, 7], [7, 7]]
      if (size === 13) return [[4, 4], [10, 4], [7, 7], [4, 10], [10, 10]]
      if (size === 19) return [[4, 4], [10, 4], [16, 4], [4, 10], [10, 10], [16, 10], [4, 16], [10, 16], [16, 16]]
      return size % 2 === 1 ? [[Math.floor(size / 2), Math.floor(size / 2)]] : []
    }

    const offset = cellSize
    const starPoints = getStarPoints(boardSize)
    ctx.fillStyle = '#808080'
    for (const [sx, sy] of starPoints) {
      const x = padding + offset + sx * cellSize
      const y = padding + offset + sy * cellSize
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, 2 * Math.PI)
      ctx.fill()
    }

    // draw stones
    const board = gameState.board
    for (let x = 0; x < boardSize; x++) {
      for (let y = 0; y < boardSize; y++) {
        if (board[x][y] !== 0) {
          const stoneX = padding + offset + x * cellSize
          const stoneY = padding + offset + y * cellSize
          const radius = cellSize * 0.4

          // draw stone
          ctx.beginPath()
          ctx.arc(stoneX, stoneY, radius, 0, 2 * Math.PI)

          if (board[x][y] === 1) {
            // black stone with gradient
            const gradient = ctx.createRadialGradient(
              stoneX - radius/3, stoneY - radius/3, 0,
              stoneX, stoneY, radius
            )
            gradient.addColorStop(0, '#555555')
            gradient.addColorStop(1, '#000000')
            ctx.fillStyle = gradient
          } else {
            // white stone with gradient
            const gradient = ctx.createRadialGradient(
              stoneX - radius/3, stoneY - radius/3, 0,
              stoneX, stoneY, radius
            )
            gradient.addColorStop(0, '#FFFFFF')
            gradient.addColorStop(1, '#CCCCCC')
            ctx.fillStyle = gradient
          }
          ctx.fill()

          // draw stone border
          ctx.strokeStyle = board[x][y] === 1 ? '#000000' : '#999999'
          ctx.lineWidth = 1
          ctx.stroke()
        }
      }
    }

    // highlight last move
    if (gameState.last_move) {
      const [lastX, lastY] = gameState.last_move
      const lastMoveX = padding + offset + lastX * cellSize
      const lastMoveY = padding + offset + lastY * cellSize

      ctx.beginPath()
      ctx.arc(lastMoveX, lastMoveY, 3, 0, 2 * Math.PI)
      ctx.fillStyle = board[lastX][lastY] === 1 ? '#FFFFFF' : '#000000'
      ctx.fill()
    }

  }, [gameState, cellSize])

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '100vh',
      backgroundColor: '#FFFFFF'
    }}>
      <canvas
        ref={canvasRef}
        style={{
          border: 'none',
          borderRadius: '0'
        }}
      />
    </div>
  )
}

export default MinimalGoBoard