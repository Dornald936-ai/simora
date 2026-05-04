const API_URL = 'http://localhost:5000/api';

let token: string | null = localStorage.getItem('simora_token');

export const setToken = (newToken: string) => {
  token = newToken;
  localStorage.setItem('simora_token', newToken);
};
export const clearToken = () => {
  token = null;
  localStorage.removeItem('simora_token');
};

const request = async (endpoint: string, options: RequestInit = {}) => {
  const headers: any = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${API_URL}${endpoint}`, { ...options, headers });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
};

export const login = (email: string, password: string) => request('/login', { method: 'POST', body: JSON.stringify({ email, password }) });
export const getSubscriptions = () => request('/subscriptions');
export const postSubscription = (data: any) => request('/subscriptions', { method: 'POST', body: JSON.stringify(data) });
export const getContacts = () => request('/contacts');
export const postContact = (data: any) => request('/contacts', { method: 'POST', body: JSON.stringify(data) });
export const getDataRequests = () => request('/data-requests');
export const postDataRequest = (data: any) => request('/data-requests', { method: 'POST', body: JSON.stringify(data) });
export const approveRequest = (id: number, days: number) => request(`/data-requests/${id}/approve`, { method: 'PUT', body: JSON.stringify({ days }) });
export const getGallery = () => request('/gallery');
export const postGallery = (data: any) => request('/gallery', { method: 'POST', body: JSON.stringify(data) });
export const getInstSubs = () => request('/inst-subscriptions');
export const postInstSub = (data: any) => request('/inst-subscriptions', { method: 'POST', body: JSON.stringify(data) });
export const getUsers = () => request('/users');
export const clearAdminData = (type: string) => request(`/admin/clear/${type}`, { method: 'DELETE' });
export const getInSAR = (lat: number, lon: number) => request(`/insar/${lat}/${lon}`);