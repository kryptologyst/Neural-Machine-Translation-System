"""
Synthetic Dataset Generator for Neural Machine Translation

This module creates synthetic datasets for testing and evaluating
translation models across different domains and complexity levels.
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationSample:
    """Represents a single translation sample."""
    source: str
    target: str
    domain: str
    complexity: str
    length: str


class SyntheticDatasetGenerator:
    """Generates synthetic translation datasets for testing."""
    
    # Pre-defined sentence templates for different domains
    TEMPLATES = {
        "greetings": {
            "en": [
                "Hello, how are you today?",
                "Good morning, nice to meet you.",
                "Have a wonderful day!",
                "Thank you for your help.",
                "See you tomorrow.",
                "How was your weekend?",
                "What's your name?",
                "Where are you from?",
                "Nice to see you again.",
                "Take care of yourself."
            ],
            "fr": [
                "Bonjour, comment allez-vous aujourd'hui ?",
                "Bonjour, ravi de vous rencontrer.",
                "Passez une merveilleuse journée !",
                "Merci pour votre aide.",
                "À demain.",
                "Comment s'est passé votre week-end ?",
                "Quel est votre nom ?",
                "D'où venez-vous ?",
                "Ravi de vous revoir.",
                "Prenez soin de vous."
            ],
            "de": [
                "Hallo, wie geht es Ihnen heute?",
                "Guten Morgen, schön Sie kennenzulernen.",
                "Haben Sie einen wundervollen Tag!",
                "Danke für Ihre Hilfe.",
                "Bis morgen.",
                "Wie war Ihr Wochenende?",
                "Wie heißen Sie?",
                "Woher kommen Sie?",
                "Schön, Sie wiederzusehen.",
                "Passen Sie auf sich auf."
            ],
            "es": [
                "Hola, ¿cómo estás hoy?",
                "Buenos días, encantado de conocerte.",
                "¡Que tengas un día maravilloso!",
                "Gracias por tu ayuda.",
                "Hasta mañana.",
                "¿Cómo fue tu fin de semana?",
                "¿Cuál es tu nombre?",
                "¿De dónde eres?",
                "Me alegra verte de nuevo.",
                "Cuídate."
            ]
        },
        "business": {
            "en": [
                "The quarterly report shows positive growth.",
                "We need to schedule a meeting for next week.",
                "The project deadline has been extended.",
                "Please review the contract before signing.",
                "Our sales team exceeded expectations this month.",
                "The budget proposal needs approval from management.",
                "We are expanding our operations to new markets.",
                "The client requested additional features.",
                "The presentation went very well.",
                "We need to hire more developers."
            ],
            "fr": [
                "Le rapport trimestriel montre une croissance positive.",
                "Nous devons planifier une réunion pour la semaine prochaine.",
                "La date limite du projet a été prolongée.",
                "Veuillez examiner le contrat avant de signer.",
                "Notre équipe de vente a dépassé les attentes ce mois-ci.",
                "La proposition de budget nécessite l'approbation de la direction.",
                "Nous étendons nos opérations à de nouveaux marchés.",
                "Le client a demandé des fonctionnalités supplémentaires.",
                "La présentation s'est très bien passée.",
                "Nous devons embaucher plus de développeurs."
            ],
            "de": [
                "Der Quartalsbericht zeigt positives Wachstum.",
                "Wir müssen ein Meeting für nächste Woche planen.",
                "Das Projektende wurde verlängert.",
                "Bitte überprüfen Sie den Vertrag vor der Unterzeichnung.",
                "Unser Verkaufsteam hat die Erwartungen diesen Monat übertroffen.",
                "Der Budgetvorschlag benötigt die Genehmigung des Managements.",
                "Wir erweitern unsere Geschäftstätigkeit auf neue Märkte.",
                "Der Kunde hat zusätzliche Funktionen angefordert.",
                "Die Präsentation lief sehr gut.",
                "Wir müssen mehr Entwickler einstellen."
            ],
            "es": [
                "El informe trimestral muestra un crecimiento positivo.",
                "Necesitamos programar una reunión para la próxima semana.",
                "La fecha límite del proyecto ha sido extendida.",
                "Por favor revise el contrato antes de firmar.",
                "Nuestro equipo de ventas superó las expectativas este mes.",
                "La propuesta de presupuesto necesita aprobación de la gerencia.",
                "Estamos expandiendo nuestras operaciones a nuevos mercados.",
                "El cliente solicitó características adicionales.",
                "La presentación fue muy bien.",
                "Necesitamos contratar más desarrolladores."
            ]
        },
        "technology": {
            "en": [
                "Machine learning algorithms are becoming more sophisticated.",
                "The neural network achieved 95% accuracy on the test set.",
                "We implemented a new data preprocessing pipeline.",
                "The API endpoint is now available for public use.",
                "The database optimization improved query performance significantly.",
                "We deployed the application to the cloud infrastructure.",
                "The code review process ensures quality and consistency.",
                "The user interface has been redesigned for better usability.",
                "We integrated third-party authentication services.",
                "The system automatically scales based on demand."
            ],
            "fr": [
                "Les algorithmes d'apprentissage automatique deviennent plus sophistiqués.",
                "Le réseau de neurones a atteint 95% de précision sur l'ensemble de test.",
                "Nous avons implémenté un nouveau pipeline de prétraitement des données.",
                "Le point de terminaison API est maintenant disponible pour un usage public.",
                "L'optimisation de la base de données a considérablement amélioré les performances des requêtes.",
                "Nous avons déployé l'application sur l'infrastructure cloud.",
                "Le processus de révision du code assure la qualité et la cohérence.",
                "L'interface utilisateur a été repensée pour une meilleure utilisabilité.",
                "Nous avons intégré des services d'authentification tiers.",
                "Le système s'adapte automatiquement en fonction de la demande."
            ],
            "de": [
                "Machine-Learning-Algorithmen werden immer ausgeklügelter.",
                "Das neuronale Netzwerk erreichte 95% Genauigkeit auf dem Testset.",
                "Wir haben eine neue Datenvorverarbeitungspipeline implementiert.",
                "Der API-Endpunkt ist jetzt für die öffentliche Nutzung verfügbar.",
                "Die Datenbankoptimierung verbesserte die Abfrageleistung erheblich.",
                "Wir haben die Anwendung auf die Cloud-Infrastruktur bereitgestellt.",
                "Der Code-Review-Prozess gewährleistet Qualität und Konsistenz.",
                "Die Benutzeroberfläche wurde für bessere Benutzerfreundlichkeit neu gestaltet.",
                "Wir haben Drittanbieter-Authentifizierungsdienste integriert.",
                "Das System skaliert automatisch basierend auf der Nachfrage."
            ],
            "es": [
                "Los algoritmos de aprendizaje automático se están volviendo más sofisticados.",
                "La red neuronal logró 95% de precisión en el conjunto de prueba.",
                "Implementamos una nueva tubería de preprocesamiento de datos.",
                "El punto final de la API ahora está disponible para uso público.",
                "La optimización de la base de datos mejoró significativamente el rendimiento de las consultas.",
                "Desplegamos la aplicación a la infraestructura en la nube.",
                "El proceso de revisión de código asegura calidad y consistencia.",
                "La interfaz de usuario ha sido rediseñada para mejor usabilidad.",
                "Integramos servicios de autenticación de terceros.",
                "El sistema se escala automáticamente basado en la demanda."
            ]
        },
        "travel": {
            "en": [
                "The hotel is located in the city center.",
                "We need to book flights for our vacation.",
                "The museum opens at nine o'clock in the morning.",
                "Can you recommend a good restaurant nearby?",
                "The train station is five minutes away on foot.",
                "We visited three countries during our trip.",
                "The weather was perfect for sightseeing.",
                "The local cuisine is absolutely delicious.",
                "We took many photographs of the beautiful scenery.",
                "The tour guide was very knowledgeable and friendly."
            ],
            "fr": [
                "L'hôtel est situé dans le centre-ville.",
                "Nous devons réserver des vols pour nos vacances.",
                "Le musée ouvre à neuf heures du matin.",
                "Pouvez-vous recommander un bon restaurant à proximité ?",
                "La gare est à cinq minutes à pied.",
                "Nous avons visité trois pays pendant notre voyage.",
                "Le temps était parfait pour le tourisme.",
                "La cuisine locale est absolument délicieuse.",
                "Nous avons pris de nombreuses photographies du magnifique paysage.",
                "Le guide touristique était très compétent et sympathique."
            ],
            "de": [
                "Das Hotel befindet sich im Stadtzentrum.",
                "Wir müssen Flüge für unseren Urlaub buchen.",
                "Das Museum öffnet um neun Uhr morgens.",
                "Können Sie ein gutes Restaurant in der Nähe empfehlen?",
                "Der Bahnhof ist fünf Minuten zu Fuß entfernt.",
                "Wir besuchten drei Länder während unserer Reise.",
                "Das Wetter war perfekt für Sightseeing.",
                "Die lokale Küche ist absolut köstlich.",
                "Wir machten viele Fotos der wunderschönen Landschaft.",
                "Der Reiseführer war sehr kenntnisreich und freundlich."
            ],
            "es": [
                "El hotel está ubicado en el centro de la ciudad.",
                "Necesitamos reservar vuelos para nuestras vacaciones.",
                "El museo abre a las nueve de la mañana.",
                "¿Puedes recomendar un buen restaurante cerca?",
                "La estación de tren está a cinco minutos caminando.",
                "Visitamos tres países durante nuestro viaje.",
                "El clima era perfecto para el turismo.",
                "La cocina local es absolutamente deliciosa.",
                "Tomamos muchas fotografías del hermoso paisaje.",
                "El guía turístico era muy conocedor y amigable."
            ]
        }
    }
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_dataset(
        self,
        language_pairs: List[Tuple[str, str]] = None,
        domains: List[str] = None,
        samples_per_domain: int = 50,
        output_file: str = "synthetic_dataset.json"
    ) -> str:
        """
        Generate a synthetic translation dataset.
        
        Args:
            language_pairs: List of (source, target) language pairs
            domains: List of domains to include
            samples_per_domain: Number of samples per domain
            output_file: Output filename
            
        Returns:
            Path to the generated dataset file
        """
        if language_pairs is None:
            language_pairs = [("en", "fr"), ("en", "de"), ("en", "es")]
        
        if domains is None:
            domains = list(self.TEMPLATES.keys())
        
        dataset = []
        
        for domain in domains:
            if domain not in self.TEMPLATES:
                logger.warning(f"Unknown domain: {domain}")
                continue
            
            for source_lang, target_lang in language_pairs:
                if source_lang not in self.TEMPLATES[domain] or target_lang not in self.TEMPLATES[domain]:
                    logger.warning(f"Language pair {source_lang}-{target_lang} not available for domain {domain}")
                    continue
                
                source_templates = self.TEMPLATES[domain][source_lang]
                target_templates = self.TEMPLATES[domain][target_lang]
                
                # Generate samples
                for i in range(samples_per_domain):
                    source_text = random.choice(source_templates)
                    target_text = random.choice(target_templates)
                    
                    # Determine complexity and length
                    complexity = self._determine_complexity(source_text)
                    length = self._determine_length(source_text)
                    
                    sample = TranslationSample(
                        source=source_text,
                        target=target_text,
                        domain=domain,
                        complexity=complexity,
                        length=length
                    )
                    
                    dataset.append({
                        "source": sample.source,
                        "target": sample.target,
                        "domain": sample.domain,
                        "complexity": sample.complexity,
                        "length": sample.length,
                        "source_lang": source_lang,
                        "target_lang": target_lang
                    })
        
        # Save dataset
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated dataset with {len(dataset)} samples: {output_path}")
        return str(output_path)
    
    def _determine_complexity(self, text: str) -> str:
        """Determine text complexity based on length and structure."""
        word_count = len(text.split())
        punctuation_count = sum(1 for c in text if c in ".,!?;:")
        
        if word_count <= 5 and punctuation_count <= 1:
            return "simple"
        elif word_count <= 15 and punctuation_count <= 3:
            return "medium"
        else:
            return "complex"
    
    def _determine_length(self, text: str) -> str:
        """Determine text length category."""
        char_count = len(text)
        
        if char_count <= 50:
            return "short"
        elif char_count <= 100:
            return "medium"
        else:
            return "long"
    
    def generate_test_set(
        self,
        language_pair: Tuple[str, str] = ("en", "fr"),
        size: int = 100
    ) -> List[Dict]:
        """
        Generate a focused test set for evaluation.
        
        Args:
            language_pair: Source and target language codes
            size: Number of test samples
            
        Returns:
            List of test samples
        """
        source_lang, target_lang = language_pair
        test_samples = []
        
        # Mix samples from all domains
        all_domains = list(self.TEMPLATES.keys())
        samples_per_domain = size // len(all_domains)
        
        for domain in all_domains:
            if source_lang in self.TEMPLATES[domain] and target_lang in self.TEMPLATES[domain]:
                source_templates = self.TEMPLATES[domain][source_lang]
                target_templates = self.TEMPLATES[domain][target_lang]
                
                for i in range(samples_per_domain):
                    source_text = random.choice(source_templates)
                    target_text = random.choice(target_templates)
                    
                    test_samples.append({
                        "source": source_text,
                        "target": target_text,
                        "domain": domain,
                        "source_lang": source_lang,
                        "target_lang": target_lang
                    })
        
        return test_samples


if __name__ == "__main__":
    # Generate synthetic dataset
    generator = SyntheticDatasetGenerator()
    
    # Generate full dataset
    dataset_path = generator.generate_dataset(
        language_pairs=[("en", "fr"), ("en", "de"), ("en", "es")],
        domains=["greetings", "business", "technology", "travel"],
        samples_per_domain=25
    )
    
    print(f"Generated dataset: {dataset_path}")
    
    # Generate test set
    test_set = generator.generate_test_set(("en", "fr"), 50)
    print(f"Generated test set with {len(test_set)} samples")
    
    # Save test set
    test_path = Path("data") / "test_set.json"
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    print(f"Test set saved to: {test_path}")
