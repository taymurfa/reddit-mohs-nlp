const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;
let pythonProcess = null;

function getPythonPath() {
    // If securely packaged, refer to the expected bundled executable location
    if (app.isPackaged) {
        const bundledExe = process.platform === 'win32' ? 'server.exe' : 'server';
        const bundledPath = path.join(process.resourcesPath, 'python-backend', bundledExe);
        if (fs.existsSync(bundledPath)) {
            return { path: bundledPath, isBundled: true };
        }
    }
    
    // In dev environment, prioritize local virtual environment if it exists
    const venvPythonPathWin = path.join(__dirname, 'venv', 'Scripts', 'python.exe');
    const venvPythonPathMac = path.join(__dirname, 'venv', 'bin', 'python');
    
    if (fs.existsSync(venvPythonPathWin)) return { path: venvPythonPathWin, isBundled: false };
    if (fs.existsSync(venvPythonPathMac)) return { path: venvPythonPathMac, isBundled: false };
    
    // Fallback to system-wide python path
    return { path: process.platform === 'win32' ? 'python' : 'python3', isBundled: false };
}

function startPythonServer() {
    const pythonConfig = getPythonPath();
    const scriptPath = path.join(__dirname, 'python', 'server.py');
    
    try {
        if (pythonConfig.isBundled) {
            pythonProcess = spawn(pythonConfig.path, []);
        } else {
            pythonProcess = spawn(pythonConfig.path, [scriptPath]);
        }
        
        pythonProcess.on('error', (err) => {
            console.error("Failed to start Python process:", err);
            dialog.showErrorBox(
                "Python Backend Not Found", 
                "Failed to start the required backend server.\n\nIf you are a developer, please ensure Python 3.10+ is installed and you created the 'venv' properly."
            );
        });
        
        pythonProcess.on('exit', (code) => {
            if (code !== 0 && code !== null) {
                console.error(`Python backend exited with code ${code}`);
            }
        });
        
    } catch (err) {
        console.error("Exception when spawning Python:", err);
    }
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        minWidth: 900,
        minHeight: 600,
        backgroundColor: '#12131a', // Prevents white flash during load
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            // Context Isolation is enabled by default which makes preload.js vital
        }
    });

    // Remove the default menu bar completely
    mainWindow.setMenuBarVisibility(false);
    mainWindow.autoHideMenuBar = true; 
    
    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
    
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Ensure Python is killed cleanly on app quit
app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});

app.whenReady().then(() => {
    startPythonServer();
    createWindow();

    // macOS doc handling
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
    
    // IPC bridge for saving CSVs directly to user's disk
    ipcMain.handle('save-csv-dialog', async (event, csvContent, defaultFilename) => {
        const result = await dialog.showSaveDialog(mainWindow, {
            title: 'Save Exported Topics',
            defaultPath: defaultFilename,
            filters: [
                { name: 'CSV File', extensions: ['csv'] }
            ]
        });
        
        if (!result.canceled && result.filePath) {
            fs.writeFileSync(result.filePath, csvContent, 'utf-8');
            return true;
        }
        return false;
    });
});

// Quit completely on Windows/Linux. On macOS, keep app dock active.
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});
