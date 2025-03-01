document.addEventListener('DOMContentLoaded', () => {
    const color1Input = document.getElementById('color1');
    const color2Input = document.getElementById('color2');
    const angleInput = document.getElementById('angle');
    const angleValue = document.getElementById('angleValue');
    const gradientPreview = document.getElementById('gradientPreview');
    const cssCode = document.getElementById('cssCode');
    const copyButton = document.getElementById('copyCode');

    function updateGradient() {
        const color1 = color1Input.value;
        const color2 = color2Input.value;
        const angle = angleInput.value;
        const gradientCSS = `linear-gradient(${angle}deg, ${color1}, ${color2})`;
        
        gradientPreview.style.background = gradientCSS;
        cssCode.textContent = `background: ${gradientCSS};`;
        angleValue.textContent = `${angle}°`;
    }

    [color1Input, color2Input, angleInput].forEach(input => {
        input.addEventListener('input', updateGradient);
    });

    copyButton.addEventListener('click', () => {
        navigator.clipboard.writeText(cssCode.textContent)
            .then(() => {
                copyButton.textContent = 'Copied!';
                setTimeout(() => {
                    copyButton.textContent = 'Copy CSS';
                }, 2000);
            });
    });

    updateGradient();
});
