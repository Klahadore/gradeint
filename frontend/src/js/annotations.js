let currentTool = null;
let annotations = [];

class Annotation {
    constructor(type, x, y) {
        this.type = type;
        this.x = x;
        this.y = y;
        this.element = null;
        this.imageUrl = this.getImageUrl(type);
    }

    getImageUrl(type) {
        switch(type) {
            case 'checkmark':
                return '../assets/correct_answer.png';
            case 'x':
                return '../assets/wrong_answer.png';
            case 'text':
                return '../assets/text_box.png';
            default:
                return null;
        }
    }

    render() {
        const annotation = document.createElement('div');
        annotation.className = `annotation ${this.type}`;
        annotation.style.left = `${this.x}px`;
        annotation.style.top = `${this.y}px`;

        const img = document.createElement('img');
        img.src = this.imageUrl;
        img.style.width = this.type === 'text' ? '100px' : '40px';
        img.style.height = 'auto';
        img.style.userSelect = 'none';
        img.draggable = false;

        annotation.appendChild(img);
        this.element = annotation;
        this.makeDraggable();
        return annotation;
    }

    makeDraggable() {
        let isDragging = false;
        let currentX;
        let currentY;

        this.element.addEventListener('mousedown', (e) => {
            if (e.target.isContentEditable && e.target === document.activeElement) return;
            isDragging = true;
            currentX = e.clientX - this.element.offsetLeft;
            currentY = e.clientY - this.element.offsetTop;
            this.element.style.cursor = 'grabbing';
        });

        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            e.preventDefault();
            this.element.style.left = `${e.clientX - currentX}px`;
            this.element.style.top = `${e.clientY - currentY}px`;
        });

        document.addEventListener('mouseup', () => {
            isDragging = false;
            this.element.style.cursor = 'grab';
        });
    }

    getPosition() {
        return {
            x: this.element.offsetLeft,
            y: this.element.offsetTop,
            type: this.type
        };
    }
}

function initAnnotations() {
    const annotationLayer = document.getElementById('annotationLayer');
    const tools = document.querySelectorAll('.tool-button');
    const saveButton = document.getElementById('saveChanges');

    tools.forEach(tool => {
        tool.addEventListener('click', () => {
            tools.forEach(t => t.classList.remove('active'));
            tool.classList.add('active');
            currentTool = tool.dataset.tool;
            annotationLayer.style.cursor = 'crosshair';
        });
    });

    annotationLayer.addEventListener('click', (e) => {
        if (!currentTool) return;

        const rect = annotationLayer.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const annotation = new Annotation(currentTool, x, y);
        annotations.push(annotation);
        annotationLayer.appendChild(annotation.render());
    });

    // Handle save changes
    saveButton.addEventListener('click', updatePreviewPDF);
}

async function updatePreviewPDF() {
    const previewFrame = document.getElementById('pdfPreview');
    const annotationPositions = annotations.map(a => a.getPosition());
    
    // Here you would typically:
    // 1. Send annotation positions to server
    // 2. Server generates new PDF with annotations
    // 3. Update preview with new PDF
    
    // For now, just update the src to simulate refresh
    previewFrame.src = previewFrame.src;
}

document.addEventListener('DOMContentLoaded', initAnnotations);
