import React from 'react';
import { render, screen } from '@testing-library/react-native';
import Header from '../src/components/header';

describe('Header Component', () => {
  test('renders correctly', () => {
    render(<Header />);
    // Aquí puedes agregar assertions específicas para tu componente
    expect(screen.getByTestId('header')).toBeTruthy();
  });

  test('displays the correct title', () => {
    const title = 'Pomodoro Timer';
    render(<Header title={title} />);
    expect(screen.getByText(title)).toBeTruthy();
  });
});
