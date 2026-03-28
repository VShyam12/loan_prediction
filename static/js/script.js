// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Handle range input display updates
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        input.addEventListener('input', function() {
            this.nextElementSibling.value = this.value;
        });
    });

    // Add animation to form elements
    const formElements = document.querySelectorAll('.form-control, .form-select, .form-check-input');
    formElements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        setTimeout(() => {
            element.style.transition = 'all 0.3s ease';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, 100 + (index * 50));
    });

    // Add hover effects to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 0.5rem 1rem rgba(0, 0, 0, 0.15)';
            this.style.transition = 'all 0.3s ease';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.1)';
        });
    });

    // Form validation
    const loanForm = document.querySelector('.loan-form');
    if (loanForm) {
        loanForm.addEventListener('submit', function(event) {
            const inputs = this.querySelectorAll('input[required], select[required]');
            let isValid = true;
            
            inputs.forEach(input => {
                if (!input.value) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!isValid) {
                event.preventDefault();
                alert('Please fill in all required fields.');
            } else {
                // Add loading state
                const submitBtn = this.querySelector('button[type="submit"]');
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                submitBtn.disabled = true;
            }
        });
    }

    // Add tooltips to risk factors and improvements
    const riskItems = document.querySelectorAll('.risk-list li');
    riskItems.forEach(item => {
        item.setAttribute('title', 'This factor increases your risk profile');
        item.style.cursor = 'help';
    });

    const improvementItems = document.querySelectorAll('.improvement-list li');
    improvementItems.forEach(item => {
        item.setAttribute('title', 'Implementing this can improve your chances');
        item.style.cursor = 'help';
    });

    // Add confetti effect for approved loans
    const resultHeader = document.querySelector('.result-header');
    if (resultHeader && resultHeader.classList.contains('approved')) {
        createConfetti();
    }
});

// Confetti animation for approved loans
function createConfetti() {
    const confettiCount = 200;
    const container = document.querySelector('body');
    
    for (let i = 0; i < confettiCount; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti';
        confetti.style.left = Math.random() * 100 + 'vw';
        confetti.style.animationDelay = Math.random() * 5 + 's';
        confetti.style.backgroundColor = getRandomColor();
        container.appendChild(confetti);
        
        // Remove confetti after animation
        setTimeout(() => {
            confetti.remove();
        }, 6000);
    }
    
    // Add confetti style
    const style = document.createElement('style');
    style.innerHTML = `
        .confetti {
            position: fixed;
            top: -10px;
            width: 10px;
            height: 10px;
            opacity: 0.7;
            z-index: 1000;
            animation: fall 5s linear forwards;
        }
        
        @keyframes fall {
            to {
                transform: translateY(100vh) rotate(720deg);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
}

function getRandomColor() {
    const colors = ['#1cc88a', '#4e73df', '#f6c23e', '#36b9cc', '#e74a3b'];
    return colors[Math.floor(Math.random() * colors.length)];
}