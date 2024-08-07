import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import {PoseDetection} from './detectPose.jsx'

function App() {
  const [isStartPose, setIsStartPose] = useState(false)
  const [modelType, setModelType] = useState(2)
  const [countRep, setCountRep] = useState(0)

  return (
    <>
      <h1>TEST MODEL KERAS</h1>
      <div
        style={{
          display : "flex",
          justifyContent : "center",
          alignItems : "center",
        }}
      >
        <PoseDetection 
          isStartPose={isStartPose}
          modelType={modelType}
          setCountRep={setCountRep}
        />
      </div>
      <div
        style={{
          display : "flex",
          justifyContent : "center",
          alignItems : "center",
          gap : "50px"
        }}
      >
        <h1
          style={{
            color: "red",
          }}
        >
          Reps : {countRep}
        </h1>
        <button onClick={() => setIsStartPose(!isStartPose)}>
          {isStartPose ? 'Stop' : 'Start'}
        </button>
        <select 
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
        >
          <option value="1">Model trả về 1 số của a Tuấn</option>
          <option value="22">Model 2 shape trả về 3 số</option>
          <option value="2">Model 2 shape trả về 2 số</option>
          <option value="3">Model 3 shape</option>
        </select>
      </div>
    </>
  )
}

export default App
