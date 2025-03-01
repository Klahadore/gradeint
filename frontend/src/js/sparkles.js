function createSparkles() {
    const container = document.querySelector('.sparkles-container');
    const sparkleCount = 75; // Increased count for more sparkles

    for (let i = 0; i < sparkleCount; i++) {
        // Create initial sparkles with staggered delays
        setTimeout(() => createSparkle(container), i * 50);
    }
}

function createSparkle(container) {
    const sparkle = document.createElement('div');
    sparkle.className = 'sparkle';
    
    // Larger size between 4px and 10px
    const size = Math.random() * 6 + 4;
    sparkle.style.width = `${size}px`;
    sparkle.style.height = `${size}px`;
    
    // Random position
    sparkle.style.left = `${Math.random() * 100}%`;
    sparkle.style.top = `${Math.random() * 100}%`;
    
    // Shorter initial delay
    sparkle.style.animationDelay = `${Math.random() * 2}s`;
    sparkle.style.animationDuration = `${Math.random() * 2 + 2}s`;
    
    container.appendChild(sparkle);
    
    sparkle.addEventListener('animationend', () => {
        sparkle.remove();
        createSparkle(container);
    });
}

// Execute immediately and on load for reliability
createSparkles();
window.addEventListener('load', createSparkles);
