const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// Function to convert YAML spec to markdown
function convertSpecToMarkdown(specPath) {
  const specContent = fs.readFileSync(specPath, 'utf8');
  const spec = yaml.load(specContent);

  // Extract frontmatter fields
  const frontmatter = {
    id: spec.chapter_id,
    title: spec.title,
    module_id: spec.module_id,
    module_title: spec.module_title,
    estimated_duration: spec.estimated_duration,
    prerequisites: spec.prerequisites
  };

  // Create markdown content
  let markdown = `---\n`;
  for (const [key, value] of Object.entries(frontmatter)) {
    if (value !== undefined) {
      if (Array.isArray(value)) {
        markdown += `${key}:\n${value.map(v => `  - "${v}"`).join('\n')}\n`;
      } else {
        markdown += `${key}: "${value}"\n`;
      }
    }
  }
  markdown += `---\n\n`;

  // Add overview
  if (spec.overview) {
    markdown += `## Overview\n\n${spec.overview}\n\n`;
  }

  // Add why it matters
  if (spec.why_it_matters) {
    markdown += `## Why It Matters\n\n${spec.why_it_matters}\n\n`;
  }

  // Add key concepts
  if (spec.key_concepts && spec.key_concepts.length > 0) {
    markdown += `## Key Concepts\n\n`;
    spec.key_concepts.forEach(concept => {
      markdown += `- ${concept.replace(/^\d+\.\s*/, '')}\n`; // Remove numbered prefixes if present
    });
    markdown += `\n`;
  }

  // Add code examples
  if (spec.code_examples && spec.code_examples.length > 0) {
    markdown += `## Code Examples\n\n`;
    spec.code_examples.forEach(example => {
      markdown += `### ${example.title}\n\n`;
      markdown += `${example.description}\n\n`;
      markdown += `**Framework:** ${example.framework || 'N/A'}\n\n`;
      markdown += `\`\`\`${example.language}\n${example.code.trim()}\n\`\`\`\n\n`;
    });
  }

  // Add practical examples
  if (spec.practical_examples && spec.practical_examples.length > 0) {
    markdown += `## Practical Examples\n\n`;
    spec.practical_examples.forEach(example => {
      markdown += `### ${example.title}\n\n`;
      markdown += `${example.description}\n\n`;

      if (example.objectives) {
        markdown += `**Objectives:**\n`;
        example.objectives.forEach(obj => markdown += `- ${obj}\n`);
        markdown += `\n`;
      }

      if (example.required_components) {
        markdown += `**Required Components:**\n`;
        example.required_components.forEach(comp => markdown += `- ${comp}\n`);
        markdown += `\n`;
      }

      if (example.evaluation_criteria) {
        markdown += `**Evaluation Criteria:**\n`;
        example.evaluation_criteria.forEach(crit => markdown += `- ${crit}\n`);
        markdown += `\n`;
      }
    });
  }

  // Add summary
  if (spec.summary) {
    markdown += `## Summary\n\n${spec.summary}\n\n`;
  }

  // Add quiz questions
  if (spec.quiz && spec.quiz.length > 0) {
    markdown += `## Quiz\n\n`;
    spec.quiz.forEach((q, index) => {
      markdown += `${index + 1}. ${q.question}\n\n`;
      q.options.forEach(option => {
        markdown += `   - ${option}\n`;
      });
      markdown += `\n   **Correct Answer:** ${q.correct_answer}\n\n`;
      markdown += `   **Explanation:** ${q.explanation}\n\n\n`;
    });
  }

  return markdown;
}

// Main function to process all spec files
function generateDocsFromSpecs() {
  const specsDir = path.join(__dirname, '..', 'specs');
  const docsDir = path.join(__dirname, '..', 'docs');

  // Create docs directory if it doesn't exist
  if (!fs.existsSync(docsDir)) {
    fs.mkdirSync(docsDir, { recursive: true });
  }

  // Read all spec files
  const specFiles = fs.readdirSync(specsDir).filter(file => file.endsWith('.yml') || file.endsWith('.yaml'));

  console.log(`Found ${specFiles.length} spec files to process...`);

  // Process each spec file
  specFiles.forEach(specFile => {
    try {
      const specPath = path.join(specsDir, specFile);
      const markdownContent = convertSpecToMarkdown(specPath);

      // Create corresponding markdown filename
      const mdFilename = specFile.replace(/\.(yml|yaml)$/, '.md');
      const mdPath = path.join(docsDir, mdFilename);

      // Write markdown file
      fs.writeFileSync(mdPath, markdownContent);
      console.log(`Generated: ${mdPath}`);
    } catch (error) {
      console.error(`Error processing ${specFile}:`, error.message);
    }
  });

  console.log('Documentation generation complete!');
}

// Run the generator
generateDocsFromSpecs();