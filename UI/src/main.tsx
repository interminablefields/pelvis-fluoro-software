import React from 'react'
import { ChakraProvider } from '@chakra-ui/react'
import theme from './theme';
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { DataProvider } from './utility/Context.tsx';
import { ColorModeScript } from '@chakra-ui/react';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ColorModeScript initialColorMode={theme.config.initialColorMode} />
    <ChakraProvider theme = {theme}>
      <DataProvider>
        <App />
      </DataProvider>
    </ChakraProvider>
  </React.StrictMode>,
)
