module.exports = {
  extends: [
    'expo',
    'prettier',
    '@react-native-community',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:jsx-a11y/recommended',
  ],
  plugins: [
    'react',
    'react-hooks',
    'react-native',
    'jsx-a11y',
    'import',
  ],
  rules: {
    // Errores comunes
    'no-unused-vars': 'error',
    'no-console': 'warn',
    'no-debugger': 'error',
    
    // React específico
    'react/prop-types': 'error',
    'react/no-unused-prop-types': 'warn',
    'react/no-array-index-key': 'warn',
    'react/jsx-key': 'error',
    'react/jsx-no-bind': 'warn',
    'react/jsx-pascal-case': 'error',
    'react/jsx-closing-bracket-location': 'error',
    'react/jsx-closing-tag-location': 'error',
    'react/jsx-curly-spacing': ['error', 'never'],
    'react/jsx-equals-spacing': ['error', 'never'],
    'react/jsx-first-prop-new-line': ['error', 'multiline'],
    'react/jsx-indent': ['error', 2],
    'react/jsx-indent-props': ['error', 2],
    'react/jsx-max-props-per-line': ['error', { maximum: 3 }],
    'react/jsx-no-duplicate-props': 'error',
    'react/jsx-no-undef': 'error',
    'react/jsx-uses-react': 'error',
    'react/jsx-uses-vars': 'error',
    'react/jsx-wrap-multilines': 'error',
    'react/self-closing-comp': 'error',
    
    // React Hooks
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',
    
    // React Native específico
    'react-native/no-unused-styles': 'warn',
    'react-native/split-platform-components': 'warn',
    'react-native/no-inline-styles': 'warn',
    'react-native/no-color-literals': 'warn',
    
    // Accesibilidad
    'jsx-a11y/accessible-emoji': 'warn',
    'jsx-a11y/alt-text': 'warn',
    
    // Importaciones
    'import/order': ['error', {
      'groups': ['builtin', 'external', 'internal', 'parent', 'sibling', 'index'],
      'newlines-between': 'always',
    }],
    'import/no-unresolved': 'error',
    'import/no-unused-modules': 'warn',
    
    // Calidad de código
    'complexity': ['warn', 10],
    'max-lines': ['warn', 300],
    'max-lines-per-function': ['warn', 50],
    'max-depth': ['warn', 4],
    'max-params': ['warn', 4],
  },
  env: {
    'react-native/react-native': true,
    es6: true,
    node: true,
    jest: true,
  },
  parserOptions: {
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: 2020,
    sourceType: 'module',
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
};
