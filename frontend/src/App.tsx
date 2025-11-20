import React, { useState, useEffect, useCallback } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box, Container, AppBar, Toolbar, Typography, IconButton, Fade } from '@mui/material';
import { Brightness4, Brightness7, Description } from '@mui/icons-material';
import { motion } from 'framer-motion';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [isUploaded, setIsUploaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Load session from localStorage on mount
  useEffect(() => {
    const savedSessionId = localStorage.getItem('mdc_session_id');
    if (savedSessionId) {
      setSessionId(savedSessionId);
      setIsUploaded(true);
    }
  }, []);

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#2563eb',
        light: '#3b82f6',
        dark: '#1d4ed8',
      },
      secondary: {
        main: '#7c3aed',
        light: '#8b5cf6',
        dark: '#6d28d9',
      },
      background: {
        default: darkMode ? '#0f172a' : '#f8fafc',
        paper: darkMode ? '#1e293b' : '#ffffff',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h4: {
        fontWeight: 700,
        letterSpacing: '-0.025em',
      },
      h6: {
        fontWeight: 600,
      },
    },
    shape: {
      borderRadius: 12,
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 600,
            borderRadius: 8,
            padding: '10px 24px',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
    },
  });

  const handleUploadSuccess = useCallback((newSessionId: string) => {
    setSessionId(newSessionId);
    setIsUploaded(true);
    localStorage.setItem('mdc_session_id', newSessionId);
  }, []);

  const handleNewSession = useCallback(() => {
    setSessionId('');
    setIsUploaded(false);
    localStorage.removeItem('mdc_session_id');
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* App Bar */}
        <AppBar 
          position="sticky" 
          elevation={0}
          sx={{ 
            bgcolor: 'background.paper',
            borderBottom: 1,
            borderColor: 'divider',
          }}
        >
          <Toolbar>
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
              style={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}
            >
              <Description sx={{ mr: 2, color: 'primary.main' }} />
              <Typography 
                variant="h6" 
                component="div" 
                sx={{ 
                  flexGrow: 1,
                  color: 'text.primary',
                  fontWeight: 700,
                }}
              >
                MultiDocChat
              </Typography>
            </motion.div>
            <IconButton 
              onClick={() => setDarkMode(!darkMode)}
              sx={{ color: 'text.primary' }}
            >
              {darkMode ? <Brightness7 /> : <Brightness4 />}
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* Main Content */}
        <Container maxWidth="lg" sx={{ py: 4 }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {!isUploaded ? (
              <FileUpload 
                onUploadSuccess={handleUploadSuccess}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
              />
            ) : (
              <Fade in={isUploaded} timeout={800}>
                <Box>
                  <ChatInterface 
                    sessionId={sessionId}
                    onNewSession={handleNewSession}
                  />
                </Box>
              </Fade>
            )}
          </motion.div>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;