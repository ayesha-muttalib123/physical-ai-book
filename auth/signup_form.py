# This file represents the signup form that would be integrated into Docusaurus
# In practice, this would be a React component in the Docusaurus src/components directory
# For this implementation, I'll provide a Python representation of how it would work

signup_form_component = """
import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './SignupForm.module.css';

function SignupForm() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    softwareBackground: '',
    hardwareBackground: '',
    contentDepth: 'standard',
    urduTranslation: false
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        alert('Account created successfully!');
        // Redirect to dashboard or next step
      } else {
        const error = await response.json();
        alert(`Error: ${error.message}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  return (
    <div className={clsx('container', styles.signupContainer)}>
      <div className={clsx('row', styles.signupRow)}>
        <div className="col col--6 col--offset-3">
          <h2>Create Your Account</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="name">Full Name</label>
              <input
                type="text"
                className="form-control"
                id="name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                className="form-control"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                className="form-control"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="softwareBackground">Software Background</label>
              <select
                className="form-control"
                id="softwareBackground"
                name="softwareBackground"
                value={formData.softwareBackground}
                onChange={handleChange}
              >
                <option value="">Select your software background</option>
                <option value="beginner">Beginner (Learning programming)</option>
                <option value="intermediate">Intermediate (Some programming experience)</option>
                <option value="advanced">Advanced (Experienced developer)</option>
                <option value="expert">Expert (Senior software engineer)</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="hardwareBackground">Hardware Background</label>
              <select
                className="form-control"
                id="hardwareBackground"
                name="hardwareBackground"
                value={formData.hardwareBackground}
                onChange={handleChange}
              >
                <option value="">Select your hardware background</option>
                <option value="beginner">Beginner (Little hardware experience)</option>
                <option value="intermediate">Intermediate (Some electronics experience)</option>
                <option value="advanced">Advanced (Experienced with hardware)</option>
                <option value="expert">Expert (Hardware engineer)</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="contentDepth">Content Depth Preference</label>
              <select
                className="form-control"
                id="contentDepth"
                name="contentDepth"
                value={formData.contentDepth}
                onChange={handleChange}
              >
                <option value="standard">Standard (Balanced content)</option>
                <option value="detailed">Detailed (More technical depth)</option>
                <option value="overview">Overview (High-level concepts)</option>
              </select>
            </div>

            <div className="form-group">
              <div className="form-check">
                <input
                  type="checkbox"
                  className="form-check-input"
                  id="urduTranslation"
                  name="urduTranslation"
                  checked={formData.urduTranslation}
                  onChange={handleChange}
                />
                <label className="form-check-label" htmlFor="urduTranslation">
                  Enable Urdu translation when available
                </label>
              </div>
            </div>

            <button type="submit" className="button button--primary button--lg">
              Create Account
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default SignupForm;
"""

# This would be the actual React component file in a real implementation
# For now, we'll just store it as a string representation
print("Signup form component would be implemented as a React component in Docusaurus")