// src/components/UrduTranslateButton.tsx
import React, { useState } from 'react';

interface UrduTranslateButtonProps {
  content: string;
  chapterId: string;
}

const UrduTranslateButton: React.FC<UrduTranslateButtonProps> = ({ content, chapterId }) => {
  const [isTranslating, setIsTranslating] = useState(false);
  const [translatedContent, setTranslatedContent] = useState<string | null>(null);
  const [showTranslation, setShowTranslation] = useState(false);

  const handleTranslate = async () => {
    setIsTranslating(true);

    try {
      // In a real implementation, this would call a translation API
      // For now, we'll simulate the translation with a placeholder
      console.log(`Translating content for chapter: ${chapterId}`);

      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));

      // This is a placeholder translation - in reality, we would call an API
      const mockTranslation = `یہ فصل ${chapterId} کا اردو میں ترجمہ ہے: \n\n${content.substring(0, 100)}... [ترجمہ مکمل]`;

      setTranslatedContent(mockTranslation);
      setShowTranslation(true);
    } catch (error) {
      console.error('Error translating content:', error);
      alert('Error translating content. Please try again.');
    } finally {
      setIsTranslating(false);
    }
  };

  const handleCopyToClipboard = () => {
    if (translatedContent) {
      navigator.clipboard.writeText(translatedContent);
      alert('Translation copied to clipboard!');
    }
  };

  return (
    <div className="urdu-translate-container">
      <button
        className="urdu-translate-btn"
        onClick={handleTranslate}
        disabled={isTranslating}
      >
        {isTranslating ? 'Translating...' : ' Urdu'}
        <span className="icon">🇺🇸→🇵🇰</span>
      </button>

      {showTranslation && translatedContent && (
        <div className="translation-panel">
          <div className="translation-header">
            <h4>اردو ترجمہ / Urdu Translation</h4>
            <button
              className="close-btn"
              onClick={() => setShowTranslation(false)}
            >
              ×
            </button>
          </div>
          <div className="translation-content">
            <p>{translatedContent}</p>
          </div>
          <div className="translation-actions">
            <button onClick={handleCopyToClipboard} className="copy-btn">
              Copy Translation
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UrduTranslateButton;