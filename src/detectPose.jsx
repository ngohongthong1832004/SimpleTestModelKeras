import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import * as tf from '@tensorflow/tfjs';

// import scaler
import scalerData from "./scalers/scaler_data.json";

// Draw line connect skeleton
const POSE_CONNECTIONS_CUSTOM = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
]


export const PoseDetection = ({
  isStartPose,
  setPercent,
  setCountRep,
  modelType
}) => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const canvasCtx = canvasRef.current?.getContext("2d")

  const poseDetection = useRef(null);
  const [modelKeras, setModelKeras] = useState(null);

  const [fps, setFps] = useState(0);
  const prevTimeRef = useRef(performance.now());
  const checkPose = [0, 0]

  const checkPose3Class = [0, 0, 0]

  const all_list_landmark = []
  const n_filter_mean = 3
  const frameCountRef = useRef(0);
  const fpsUpdateInterval = 1000; // Update FPS every second
  let rafId = useRef(null);

  const urlModelKeras = import.meta.env.BASE_URL + "public/models/model.json";

  // console.log("scaler", scaler)

  // Load model
  useEffect(() => {
    try {
      const setBackendTf = async () => {
        await tf.setBackend('webgl');
      }
      setBackendTf
      // load pose detection
      const runPoseDetection = async () => {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        poseDetection.current = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            // modelAssetPath: ModelCheckPose,
            // modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU",
          },
          // outputCategoryMask : true,
          // outputSegmentationMasks: true,
          runningMode: "VIDEO",
          numPoses: 1,
        });
      };
      runPoseDetection();
    } catch (error) {
      console.log("error", error);
    }
  }, []);

  useEffect(() => {
    const getModelKeras = async () => {
      try {
        console.log('Loading modelKeras...');
        const model = await tf.loadLayersModel(urlModelKeras);
        setModelKeras(model)
        console.log('ModelKeras loaded successfully');
      } catch (error) {
        console.error('Error loading model', error);
      }
    }

    getModelKeras();
  }, []);

  // Function to draw connectors
  function customDrawConnectors(ctx, landmarks, connections) {
    try {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;

      for (const connection of connections) {
        const [startIdx, endIdx] = connection;
        const startLandmark = landmarks[startIdx];
        const endLandmark = landmarks[endIdx];

        ctx.beginPath();
        ctx.moveTo(startLandmark?.x * ctx.canvas.width, startLandmark?.y * ctx.canvas.height);
        ctx.lineTo(endLandmark?.x * ctx.canvas.width, endLandmark?.y * ctx.canvas.height);
        ctx.stroke();
      }
    }
    catch (e) {
      // console.log("error", e)
    }
  }

  // Function to get result from the model
  async function onResults(results) {
    try {
      const landmarks = results?.landmarks?.[0];

      let landmarkFilterMean = [];
      all_list_landmark.push(landmarks);
      if (all_list_landmark.length > n_filter_mean) {
        all_list_landmark.shift();
      }
      if (all_list_landmark.length > 0) {
        for (let key in all_list_landmark[0]) {
          landmarkFilterMean[key] = {
            x: 0,
            y: 0,
            z: 0,
            visibility: 0
          };
        }
        let n = all_list_landmark.length;
        // Calculate the sum of each landmark
        for (let landmarks of all_list_landmark) {
          for (let key in landmarks) {
            landmarkFilterMean[key]["x"] += landmarks?.[key]?.x;
            landmarkFilterMean[key]["y"] += landmarks?.[key]?.y;
            landmarkFilterMean[key]["z"] += landmarks?.[key]?.z;
            landmarkFilterMean[key]["visibility"] += landmarks?.[key]?.visibility;
          }
        }
        // Calculate the average
        for (let key in landmarkFilterMean) {
          landmarkFilterMean[key]["x"] /= n;
          landmarkFilterMean[key]["y"] /= n;
          landmarkFilterMean[key]["z"] /= n;
          landmarkFilterMean[key]["visibility"] /= n;
        }
      }

      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      for (const landmark of [landmarks]) {
        customDrawConnectors(canvasCtx, landmark, POSE_CONNECTIONS_CUSTOM);
      }
      canvasCtx.save();


      const xList = landmarkFilterMean?.map((item) => item?.x || 0) || [];
      const yList = landmarkFilterMean?.map((item) => item?.y || 0) || [];
      const zList = landmarkFilterMean?.map((item) => item?.z || 0) || [];
      const vList = landmarkFilterMean?.map((item) => item?.visibility || 0) || [];

      const normalizeData = (data, scaler, indices) => {
        const mean = scaler.mean_values;
        const variance = scaler.var_values;
        const scaledData = data.map((val, idx) => (val - mean[idx]) / Math.sqrt(variance[idx]));
        return tf.tensor2d([scaledData], [1, indices.length * 4]);
      };

      if (modelKeras) {
        function getElementsByIndices(array, indices) {
          return indices.map(index => array[index]);
        }


        // model trả về 2 số
        if (modelType == 2) {
          const indices = [0, 11, 12, 14, 13, 16, 15, 23, 24];
          const combinedList = [];
          for (let i = 0; i < indices.length; i++) {
            combinedList.push(getElementsByIndices(xList, indices)[i]);
            combinedList.push(getElementsByIndices(yList, indices)[i]);
            combinedList.push(getElementsByIndices(zList, indices)[i]);
            combinedList.push(getElementsByIndices(vList, indices)[i]);
          }
          // console.log("trainData", trainData)
          // const dummyData = tf.tensor2d(trainData, [trainShape1, indices.length * 4]);

          const dummyData = normalizeData(
            combinedList,
            scalerData,
            indices
          );

          const preds = await modelKeras.predict(dummyData);
          await preds.array().then((predictions) => {
            console.log("predictions", predictions[0])
            function findMaxElementAndIndex(arr) {
              let maxElement = 0;
              let maxIndex = 0;
              for (let i = 0; i < arr.length; i++) {
                if (arr[i] > maxElement && arr[i] > 0.6) {
                  maxElement = arr[i];
                  maxIndex = i;
                }
              }
              return { maxElement, maxIndex };
            }
            const maxPredictions = findMaxElementAndIndex(predictions[0])
            if (maxPredictions.maxIndex == 0) {
              checkPose[0] = 1;
            } else if (maxPredictions.maxIndex == 1 && checkPose[0] == 1) {
              checkPose[1] = 1;
            }
            if (checkPose.every((item) => item === 1)) {
              for (let i = 0; i < checkPose.length; i++) {
                checkPose[i] = 0;
              }
              setCountRep((prev) => prev + 1);
            }
          });
          return
        }

        // model trả về 3 số
        else if (modelType == 3) {
          // "NOSE",
          // "LEFT_SHOULDER",
          // "RIGHT_SHOULDER",
          // "LEFT_ELBOW",
          // "RIGHT_ELBOW",
          // "LEFT_WRIST",
          // "RIGHT_WRIST",
          // "LEFT_HIP",
          // "RIGHT_HIP",
          // "LEFT_KNEE",
          // "RIGHT_KNEE",
          // "LEFT_ANKLE",
          // "RIGHT_ANKLE",
          // "LEFT_HEEL",
          // "RIGHT_HEEL",
          // "LEFT_FOOT_INDEX",
          // "RIGHT_FOOT_INDEX",
          const indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
          const combinedList = [];
          // console.log(getElementsByIndices(xList, indices))
          for (let i = 0; i < indices.length; i++) {
            combinedList.push(getElementsByIndices(xList, indices)[i]);
            combinedList.push(getElementsByIndices(yList, indices)[i]);
            combinedList.push(getElementsByIndices(zList, indices)[i]);
            combinedList.push(getElementsByIndices(vList, indices)[i]);
          }

          const normalizeData = (data, scaler, indices) => {
            const mean = scaler.mean_values;
            const variance = scaler.var_values;
            const scaledData = data.map((val, idx) => (val - mean[idx]) / Math.sqrt(variance[idx]));
            return tf.tensor3d([scaledData.map(a => [a])], [1, indices.length * 4, 1]);
          };

          const dummyData = normalizeData(
            combinedList,
            scalerData,
            indices
          );

          // console.log("dummyData", dummyData)

          // const dummyData = tf.tensor2d(trainData, [trainShape1, trainShape2]);
          const preds = await modelKeras.predict(dummyData);
          await preds.array().then((predictions) => {
            console.log("predictions", predictions[0])
            function findMaxElementAndIndex(arr) {
              let maxElement = 0;
              let maxIndex = 0;
              for (let i = 0; i < arr.length; i++) {
                if (arr[i] > maxElement && arr[i] > 0.90) {
                  maxElement = arr[i];
                  maxIndex = i;
                }
              }
              return { maxElement, maxIndex };
            }
            const maxPredictions = findMaxElementAndIndex(predictions[0])
            if (maxPredictions.maxIndex == 0) {
              checkPose[0] = 1;
            } else if (maxPredictions.maxIndex == 1 && checkPose[0] == 1) {
              checkPose[1] = 1;
            }
            if (checkPose.every((item) => item === 1)) {
              for (let i = 0; i < checkPose.length; i++) {
                checkPose[i] = 0;
              }
              setCountRep((prev) => prev + 1);
            }
          });
          return
        }

        const trainData = [...xList, ...yList]
        const trainShape2 = 66 // 
        const dummyData = tf.tensor2d([trainData], [1, trainShape2]);
        const preds = await modelKeras.predict(dummyData);
        await preds.array().then((predictions) => {
          console.log(predictions[0])
          if (predictions[0] != 1) {
            setPercent((predictions[0] * 100));
          }
          const maxPredictions = predictions[0] >= 0.6 ? 1 : predictions[0] <= 0.2 ? 0 : -1;
          if (maxPredictions == 0) {
            checkPose[0] = 1;
          } else if (maxPredictions == 1 && checkPose[0] == 1) {
            checkPose[1] = 1;
          }
          if (checkPose.every((item) => item === 1)) {
            for (let i = 0; i < checkPose.length; i++) {
              checkPose[i] = 0;
            }
            setCountRep((prev) => prev + 1);
          }
        });
      }
    } catch (err) {
      // Bỏ đi cho clean
      console.log("err", err);
    }
  }

  const updateFPS = async () => {
    const currentTimeMs = performance.now();
    const timeElapsed = currentTimeMs - prevTimeRef.current;

    frameCountRef.current += 1;

    // Update FPS every second
    if (timeElapsed >= fpsUpdateInterval) {
      const newFps = Math.round((frameCountRef.current / timeElapsed) * 1000);
      setFps(newFps);
      frameCountRef.current = 0;
      prevTimeRef.current = currentTimeMs;
    }

    try {
      const startTimeMs = performance.now();
      poseDetection.current.detectForVideo(
        webcamRef.current.video,
        startTimeMs,
        (result) => {
          onResults(result);
        }
      );
    } catch (error) {
      console.log("Error estimating poses:", error);
    }

    // Request the next frame
    rafId.current = requestAnimationFrame(updateFPS);
  };


  useEffect(() => {
    if (isStartPose) {
      rafId.current = requestAnimationFrame(updateFPS);

    } else {
      if (rafId.current) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
    }

    return () => {
      // Clean up on unmount or when startRecord changes
      if (rafId.current) {
        cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
    };
  }, [isStartPose, modelType]);

  return (
    <div
      style={{
        position: "relative",
        left: 0,
        top: 0,
        fontSize: "20px",
        color: "red",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          fontSize: "20px",
          color: "red",
          fontWeight: "bold",
          zIndex: 1000,
        }}
      >
        FPS: {fps}
      </div>
      <Webcam
        className="w-full h-full object-cover rounded-[16px]"
        id="webcam"
        ref={webcamRef}
        mirrored={false}
        imageSmoothing
      />
      <canvas
        ref={canvasRef}
        width="640px"
        height="480px"
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          right: 0,
          bottom: 0,
          zIndex: 35,
        }}
      ></canvas>
    </div>
  );
};
