// src/components/auth/SignupForm.tsx
import React, { useState } from 'react';
import authService, { RegistrationData } from './authService';

interface SignupFormProps {
  onSignupSuccess: (userData: any) => void;
}

const SignupForm: React.FC<SignupFormProps> = ({ onSignupSuccess }) => {
  const [formData, setFormData] = useState({
    email: '',
    name: '',
    password: '',
    is_software_focused: null as boolean | null,
    learning_path: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleFocusChange = (value: boolean) => {
    setFormData(prev => ({
      ...prev,
      is_software_focused: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const registrationData: RegistrationData = {
        email: formData.email,
        name: formData.name,
        auth_method: 'email',
        is_software_focused: formData.is_software_focused,
        learning_path: formData.learning_path
      };

      const result = await authService.register(registrationData);
      onSignupSuccess(result);
    } catch (err) {
      setError('Signup failed. Please try again.');
      console.error('Signup error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="signup-form-container">
      <h2>Create Account</h2>
      {error && <div className="alert alert-danger">{error}</div>}

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="name">Full Name</label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            required
            className="form-control"
          />
        </div>

        <div className="form-group">
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            required
            className="form-control"
          />
        </div>

        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            required
            className="form-control"
          />
        </div>

        <div className="form-group">
          <label>Which area interests you more?</label>
          <div className="radio-group">
            <label>
              <input
                type="radio"
                name="focus"
                checked={formData.is_software_focused === true}
                onChange={() => handleFocusChange(true)}
              />
              Software & Algorithms
            </label>
            <label>
              <input
                type="radio"
                name="focus"
                checked={formData.is_software_focused === false}
                onChange={() => handleFocusChange(false)}
              />
              Hardware & Mechanics
            </label>
            <label>
              <input
                type="radio"
                name="focus"
                checked={formData.is_software_focused === null}
                onChange={() => handleFocusChange(null)}
              />
              Both Equally
            </label>
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="learning_path">Learning Path</label>
          <select
            id="learning_path"
            name="learning_path"
            value={formData.learning_path}
            onChange={handleChange}
            className="form-control"
          >
            <option value="">Select your learning path</option>
            <option value="beginner">Beginner: Start with basics</option>
            <option value="intermediate">Intermediate: Have some experience</option>
            <option value="advanced">Advanced: Experienced practitioner</option>
            <option value="researcher">Researcher: Academic focus</option>
            <option value="engineer">Engineer: Practical applications</option>
          </select>
        </div>

        <button type="submit" disabled={loading} className="btn btn-primary">
          {loading ? 'Creating Account...' : 'Sign Up'}
        </button>
      </form>
    </div>
  );
};

export default SignupForm;