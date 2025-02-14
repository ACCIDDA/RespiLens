export const getDataPath = (path) => {
  const baseUrl = import.meta.env.BASE_URL || '';
  // Remove trailing slash from baseUrl if it exists
  const cleanBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
  return `${cleanBaseUrl}/processed_data/${path}`;
}; 