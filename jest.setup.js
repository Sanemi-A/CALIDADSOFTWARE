import 'react-native-gesture-handler/jestSetup';

// Mock de AsyncStorage
jest.mock('@react-native-async-storage/async-storage', () =>
  require('@react-native-async-storage/async-storage/jest/async-storage-mock')
);

// Mock de expo-av
jest.mock('expo-av', () => ({
  Audio: {
    Sound: {
      createAsync: jest.fn(() =>
        Promise.resolve({
          playAsync: jest.fn(),
          stopAsync: jest.fn(),
          unloadAsync: jest.fn(),
        })
      ),
    },
    setAudioModeAsync: jest.fn(),
  },
}));

// Mock de Expo modules
jest.mock('expo-constants', () => ({
  default: {
    expoConfig: {
      name: 'Pomodoro Timer',
      version: '1.0.0',
    },
  },
}));

// Configuración global de timeout para tests
jest.setTimeout(10000);

// Mock de console warnings para tests más limpios
const originalWarn = console.warn;
beforeAll(() => {
  console.warn = (...args) => {
    if (
      typeof args[0] === 'string' &&
      args[0].includes('componentWillReceiveProps')
    ) {
      return;
    }
    originalWarn.call(console, ...args);
  };
});

afterAll(() => {
  console.warn = originalWarn;
});
