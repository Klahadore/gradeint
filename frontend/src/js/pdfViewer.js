// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = '/lib/pdfjs/pdf.worker.js';

class PDFViewer {
    constructor() {
        this.pdfDoc = null;
        this.pageNum = 1;
        this.pageCount = 0;
        this.canvas = document.getElementById('pdfViewer');
        this.ctx = this.canvas.getContext('2d');
        this.container = document.getElementById('pdfContainer');
        
        // Page navigation controls
        this.prevButton = document.getElementById('prevPage');
        this.nextButton = document.getElementById('nextPage');
        this.pageNumSpan = document.getElementById('pageNum');
        this.pageCountSpan = document.getElementById('pageCount');

        this.initializeControls();
        this.loadPDF();
    }

    initializeControls() {
        this.prevButton.addEventListener('click', () => this.previousPage());
        this.nextButton.addEventListener('click', () => this.nextPage());
    }

    async loadPDF() {
        try {
            console.log('Starting PDF load');
            // Use absolute path from server root
            const url = '/assets/mcq_42.pdf';
            
            // Show loading state
            this.container.innerHTML = '<div style="padding: 20px;">Loading PDF...</div>';
            
            // Create canvas if needed
            if (!this.canvas) {
                this.canvas = document.createElement('canvas');
                this.canvas.id = 'pdfViewer';
                this.ctx = this.canvas.getContext('2d');
            }
            
            // Load document with better error handling
            const loadingTask = pdfjsLib.getDocument({
                url: url,
                withCredentials: false,
                cMapUrl: 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.4.120/cmaps/',
                cMapPacked: true,
            });
            
            this.pdfDoc = await loadingTask.promise;
            this.pageCount = this.pdfDoc.numPages;
            this.pageCountSpan.textContent = this.pageCount;
            console.log('PDF loaded successfully');
            
            // Clear container and add canvas
            this.container.innerHTML = '';
            this.container.appendChild(this.canvas);
            
            await this.renderPage(this.pageNum);
            
        } catch (error) {
            console.error('PDF Error:', error);
            this.container.innerHTML = `
                <div style="color: red; padding: 20px;">
                    Error loading PDF:<br>
                    ${error.message}<br><br>
                    Make sure you're running a local server<br>
                    and accessing the page via http://localhost:8000
                </div>`;
        }
    }

    async renderPage(num) {
        try {
            console.log('Rendering page', num);
            this.pageNum = num;
            this.pageNumSpan.textContent = num;
            
            const page = await this.pdfDoc.getPage(num);
            
            // Calculate scale to fit container width
            const containerWidth = this.container.clientWidth - 40; // Account for padding
            const viewport = page.getViewport({ scale: 1.0 });
            const scale = containerWidth / viewport.width;
            const scaledViewport = page.getViewport({ scale });

            // Set canvas size
            this.canvas.width = scaledViewport.width;
            this.canvas.height = scaledViewport.height;

            // Render PDF page
            const renderContext = {
                canvasContext: this.ctx,
                viewport: scaledViewport,
                background: 'rgba(0,0,0,0)'
            };

            await page.render(renderContext).promise;
            console.log('Page rendered successfully');

            // Update navigation buttons
            this.prevButton.disabled = this.pageNum <= 1;
            this.nextButton.disabled = this.pageNum >= this.pageCount;

        } catch (error) {
            console.error('Error rendering page:', error);
            this.container.innerHTML = `
                <div style="color: red; padding: 20px;">
                    Error rendering page:<br>
                    ${error.message}
                </div>`;
        }
    }

    previousPage() {
        if (this.pageNum <= 1) return;
        this.renderPage(this.pageNum - 1);
    }

    nextPage() {
        if (this.pageNum >= this.pageCount) return;
        this.renderPage(this.pageNum + 1);
    }
}

class PDFManager {
    constructor() {
        this.pdfViewer = new PDFViewer();
        this.currentTab = 'edit';
        this.annotations = [];
        this.initializeTabs();
        this.initializeSaveButton();
    }

    initializeTabs() {
        const tabs = document.querySelectorAll('.tab-button');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => this.switchTab(tab.dataset.tab));
        });
    }

    initializeSaveButton() {
        const saveButton = document.getElementById('saveChanges');
        saveButton.addEventListener('click', () => this.updatePreviewPDF());
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabName);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}Tab`);
        });

        this.currentTab = tabName;
        
        if (tabName === 'preview') {
            this.updatePreviewPDF();
        }
    }

    async updatePreviewPDF() {
        const previewFrame = document.getElementById('pdfPreview');
        // Here you would typically:
        // 1. Generate a new PDF with annotations
        // 2. Update the preview iframe src
        // For now, we'll just update the src to the original PDF
        previewFrame.src = '/assets/mcq_42.pdf';
    }
}

// Initialize PDFManager instead of PDFViewer directly
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new PDFManager());
} else {
    new PDFManager();
}
