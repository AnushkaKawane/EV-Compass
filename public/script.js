document.addEventListener("DOMContentLoaded", () => {

    // Prevent accidental submission with invalid forms.
    const forms = document.querySelectorAll("form");

    forms.forEach((form) => {

        form.addEventListener("submit", (event) => {

            if (!form.checkValidity()) {
                event.preventDefault();

                form.reportValidity();
            }

        });

    });


    // Add a subtle loading state to the assessment button.
    const assessmentForm = document.querySelector(
        'form[action*="/assess"]'
    );

    if (assessmentForm) {

        assessmentForm.addEventListener("submit", () => {

            if (!assessmentForm.checkValidity()) {
                return;
            }

            const button = assessmentForm.querySelector(
                'button[type="submit"]'
            );

            if (button) {

                button.textContent = "Analysing your profile...";

                button.disabled = true;

                button.style.opacity = "0.75";

            }

        });

    }


    // Smoothly reveal cards as they enter the viewport.
    const cards = document.querySelectorAll(
        ".feature-card, .result-card, .stat-card"
    );

    if (
        "IntersectionObserver" in window &&
        !window.matchMedia(
            "(prefers-reduced-motion: reduce)"
        ).matches
    ) {

        const observer = new IntersectionObserver(
            (entries) => {

                entries.forEach((entry) => {

                    if (entry.isIntersecting) {

                        entry.target.classList.add(
                            "is-visible"
                        );

                        observer.unobserve(
                            entry.target
                        );

                    }

                });

            },
            {
                threshold: 0.08
            }
        );

        cards.forEach((card) => {

            card.style.opacity = "0";

            card.style.transform = "translateY(8px)";

            card.style.transition =
                "opacity 0.35s ease, transform 0.35s ease";

            observer.observe(card);

        });

        const revealStyle = document.createElement("style");

        revealStyle.textContent = `
            .is-visible {
                opacity: 1 !important;
                transform: translateY(0) !important;
            }
        `;

        document.head.appendChild(revealStyle);

    }

});