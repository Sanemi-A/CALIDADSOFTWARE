import { StyleSheet, Text, View } from "react-native";


export default function Timer({time}){
    const formatTime = `${Math.floor(time / 60).toString().padStart(2,"0")}:${(time%60).toString().padStart(2,"0")}`;
    return (
        <View style={styles.container}>
            <Text style={styles.time}>{formatTime}</Text>
        </View>
    )
    
}


const styles = StyleSheet.create({
    container: {
        flex:0.3,
        justifyContent: "center",
        backgroundColor: "#F2F2F2",
        padding: 15,
        borderRadius: 15
    },
    time: {
        fontWeight: "bold",
        fontSize: 80,
        textAlign: "center",
        color: "#333333"
    }
})
