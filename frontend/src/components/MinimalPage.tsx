import { useState, useEffect, useRef } from 'react'
import MinimalGoBoard from './MinimalGoBoard'

interface GameState {
  board: number[][]
  current_player: number
  move_count: number
  last_move: [number, number] | null
  captured_black: number
  captured_white: number
  game_over: boolean
}

const MinimalPage = () => {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        // use relative URL to work with both development and production
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => {
          console.log('WebSocket connected')
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            if (data.type === 'game_update' || data.type === 'ping') {
              if (data.type === 'game_update') {
                setGameState(data)
              }
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }

        ws.onclose = () => {
          console.log('WebSocket disconnected')
          // try to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000)
        }

        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          ws.close()
        }
      } catch (error) {
        console.error('Failed to connect WebSocket:', error)
        // try again after 5 seconds
        setTimeout(connectWebSocket, 5000)
      }
    }

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  if (!gameState) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        backgroundColor: '#FFFFFF',
        color: '#666666',
        fontSize: '18px'
      }}>
        Loading game state...
      </div>
    )
  }

  return <MinimalGoBoard gameState={gameState} />
}

export default MinimalPage