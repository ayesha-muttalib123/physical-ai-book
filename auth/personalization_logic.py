from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalizationEngine:
    """
    Engine for handling content personalization based on user profiles
    """
    def __init__(self):
        self.content_levels = {
            "standard": 1.0,      # Default level
            "detailed": 1.5,      # More technical depth
            "overview": 0.7       # High-level overview
        }

    def adjust_content_for_user(self, content: str, user_profile: Dict[str, Any]) -> str:
        """
        Adjust content based on user profile
        """
        if not user_profile:
            return content

        # Apply content level adjustment
        content_level = user_profile.get("content_depth_preference", "standard")
        adjusted_content = self._apply_content_level(content, content_level)

        # Apply background-specific additions
        adjusted_content = self._apply_background_specific_content(
            adjusted_content, user_profile
        )

        # Apply language preferences
        if user_profile.get("urdu_translation_enabled", False):
            adjusted_content = self._add_urdu_translation(adjusted_content)

        return adjusted_content

    def _apply_content_level(self, content: str, level: str) -> str:
        """
        Apply content level adjustments
        """
        if level == "detailed":
            # Add more technical details for detailed level
            content += "\n\n**Technical Deep Dive:** For users seeking more depth, this concept involves advanced algorithms and mathematical principles that are crucial for implementation."
        elif level == "overview":
            # Simplify content for overview level
            content = self._simplify_content(content)

        return content

    def _apply_background_specific_content(self, content: str, user_profile: Dict[str, Any]) -> str:
        """
        Add content based on user's background
        """
        software_bg = user_profile.get("software_background", "").lower()
        hardware_bg = user_profile.get("hardware_background", "").lower()

        # Add software-focused content for users with software background
        if any(bg in software_bg for bg in ["software", "computer", "programming", "developer", "engineer"]):
            content += "\n\n> **Software Engineer Note:** This concept is particularly relevant when implementing robotic systems in software. Consider how this algorithm would be implemented in your preferred programming language."

        # Add hardware-focused content for users with hardware background
        if any(bg in hardware_bg for bg in ["hardware", "electronics", "mechanical", "electrical", "robotics"]):
            content += "\n\n> **Hardware Engineer Note:** When implementing this concept in hardware, consider the physical constraints and real-world limitations that might affect performance."

        return content

    def _add_urdu_translation(self, content: str) -> str:
        """
        Add Urdu translation where applicable
        """
        # This is a placeholder - in a real implementation, you'd have actual Urdu translations
        urdu_equivalent = {
            "Physical AI": "مادی مصنوعی ذہانت",
            "Robotics": "روبوٹکس",
            "Embodied Intelligence": "جسمانی ذہانت",
            "Humanoid Robotics": "انسان نما روبوٹکس"
        }

        translated_content = content
        for english_term, urdu_term in urdu_equivalent.items():
            translated_content = translated_content.replace(
                english_term, f"{english_term} ({urdu_term})"
            )

        return translated_content

    def _simplify_content(self, content: str) -> str:
        """
        Simplify content for overview level
        """
        # Remove technical deep dive sections
        simplified = content.replace(
            "\n\n**Technical Deep Dive:**",
            "\n\n**Key Point:**"
        )

        # Simplify complex explanations
        return simplified

    def get_personalized_chapter_content(self, chapter_id: str, user_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get personalized content for a specific chapter
        """
        # In a real implementation, this would fetch the actual chapter content
        # For now, we'll create mock content
        base_content = self._get_chapter_content(chapter_id)

        if user_profile:
            personalized_content = self.adjust_content_for_user(base_content, user_profile)
        else:
            personalized_content = base_content

        return {
            "chapter_id": chapter_id,
            "content": personalized_content,
            "personalization_applied": user_profile is not None,
            "urdu_translation_enabled": user_profile.get("urdu_translation_enabled", False) if user_profile else False
        }

    def _get_chapter_content(self, chapter_id: str) -> str:
        """
        Get base content for a chapter (in a real implementation, this would fetch from the textbook)
        """
        # Mock content for demonstration
        chapter_contents = {
            "00-introduction": "# Introduction to Physical AI\n\nPhysical AI represents a paradigm shift from traditional digital AI to embodied intelligence systems...",
            "01-physical-ai-basics": "# Physical AI Basics\n\nPhysical AI combines artificial intelligence with physical systems to interact with the real world...",
            "02-sensing-and-embodied-intelligence": "# Sensing and Embodied Intelligence\n\nEmbodied intelligence emerges from the interaction between an agent and its environment...",
            "03-ros2-nervous-system": "# ROS2 as the Nervous System\n\nThe Robot Operating System (ROS2) serves as the communication backbone for robotic applications...",
            "default": f"Content for chapter {chapter_id}"
        }

        return chapter_contents.get(chapter_id, chapter_contents["default"])

# Example usage
if __name__ == "__main__":
    engine = PersonalizationEngine()

    # Example user profile
    user_profile = {
        "software_background": "advanced",
        "hardware_background": "intermediate",
        "content_depth_preference": "detailed",
        "urdu_translation_enabled": True
    }

    # Get personalized content
    result = engine.get_personalized_chapter_content("01-physical-ai-basics", user_profile)
    print("Personalized Content:")
    print(result["content"])
    print(f"Personalization applied: {result['personalization_applied']}")
    print(f"Urdu translation enabled: {result['urdu_translation_enabled']}")