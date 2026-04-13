const { contextBridge, ipcRenderer } = require('electron');

// Expose safe methods and IPC triggers locally to window.electronAPI
contextBridge.exposeInMainWorld('electronAPI', {
    saveCsvDialog: (csvContent, defaultFilename) => ipcRenderer.invoke('save-csv-dialog', csvContent, defaultFilename)
});
