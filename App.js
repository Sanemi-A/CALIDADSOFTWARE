import { StatusBar } from "expo-status-bar";
import {
  StyleSheet,
  Text,
  View,
  Button,
  SafeAreaView,
  Platform,
  TouchableOpacity,
} from "react-native";
import { useEffect, useState } from "react";
import Header from "./src/components/header";
import Timer from "./src/components/timer";
import { Audio } from "expo-av";

const colors = ["#F7DC6F", "#A2D9CE", "#D7BDE2"];

export default function App() {
  const [isWorking, setIsWorking] = useState(false);
  const [time, setTime] = useState(25 * 60);
  const [currentTime, setCurrentTime] = useState("POMO" | "SHORT" | "BREAK");
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    let interval = null;

    if (isActive) {
      interval = setInterval(() => {
        setTime(time - 1);
      }, 1000);
    } else {
      clearInterval(interval);
    }

    if (time === 0) {
      setIsActive(false);
      setIsWorking((prev) => !prev);
      setTime(isWorking ? 300 : 1500);
    }

    return () => clearInterval(interval);
  }, [isActive, time]);

  useEffect(() => {
  setIsActive(false);
}, [currentTime]);


  function handleStartStop() {
    playSound();
    setIsActive(!isActive);
  }

  async function playSound() {
    const { sound } = await Audio.Sound.createAsync(
      require("./assets/bambu_1.mp3")
    );
    await sound.playAsync();
  }

  return (
    <SafeAreaView
      style={[styles.container, { backgroundColor: colors[currentTime] }]}
    >
      <View style={styles.view}>
        <Text style={styles.text}>Pomodoro</Text>
        <Header
          currentTime={currentTime}
          setTime={setTime}
          setCurrentTime={setCurrentTime}
        />
        <Timer time={time} />
        <TouchableOpacity onPress={handleStartStop} style={styles.button}>
          <Text style={styles.start}>{isActive ? "STOP" : "START"}</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  text: {
    fontSize: 32,
    fontWeight: "bold",
  },
  view: {
    flex: 1,
    paddingTop: Platform.OS == "android" ? 30 : 0,
    paddingHorizontal: 15,
  },
  start: {
    color: "white",
    fontWeight: "bold",
  },
  button: {
    backgroundColor: "#333333",
    padding: 15,
    borderRadius: 15,
    alignItems: "center",
    marginTop: 25,
  },
});
