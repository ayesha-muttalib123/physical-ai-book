// src/components/PersonalizeButton.tsx
import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface PersonalizeButtonProps {
  chapterId: string;
  chapterTitle: string;
}

const PersonalizeButton: React.FC<PersonalizeButtonProps> = ({ chapterId, chapterTitle }) => {
  const { user } = useAuth();
  const [isPersonalizing, setIsPersonalizing] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const [selectedOption, setSelectedOption] = useState('default');

  const handlePersonalize = async () => {
    if (!user) {
      alert('Please login to personalize content');
      return;
    }

    setIsPersonalizing(true);

    try {
      // In a real implementation, this would call the backend API
      // to get personalized content based on user preferences
      console.log(`Personalizing chapter: ${chapterId} for user: ${user.id}`);

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Apply personalization based on user preferences
      alert(`Content for chapter "${chapterTitle}" has been personalized based on your preferences!`);
    } catch (error) {
      console.error('Error personalizing content:', error);
      alert('Error personalizing content. Please try again.');
    } finally {
      setIsPersonalizing(false);
      setShowOptions(false);
    }
  };

  const personalizationOptions = [
    { value: 'beginner', label: 'Beginner-friendly' },
    { value: 'advanced', label: 'Advanced concepts' },
    { value: 'software-focused', label: 'Software-focused' },
    { value: 'hardware-focused', label: 'Hardware-focused' },
    { value: 'examples-heavy', label: 'Examples-heavy' },
    { value: 'theory-heavy', label: 'Theory-heavy' },
  ];

  return (
    <div className="personalize-container">
      <button
        className="personalize-btn"
        onClick={() => user ? setShowOptions(!showOptions) : alert('Please login to personalize content')}
        disabled={isPersonalizing}
      >
        {isPersonalizing ? 'Personalizing...' : ' personalize'}
        <span className="icon">🎯</span>
      </button>

      {showOptions && (
        <div className="personalize-options">
          <div className="option-group">
            <label>Complexity Level:</label>
            <select
              value={selectedOption}
              onChange={(e) => setSelectedOption(e.target.value)}
            >
              {personalizationOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <button
            className="apply-personalization-btn"
            onClick={handlePersonalize}
            disabled={isPersonalizing}
          >
            Apply Personalization
          </button>
        </div>
      )}
    </div>
  );
};

export default PersonalizeButton;