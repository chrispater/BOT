import axios from 'axios'

export const API_URL = '/api'

export function createApi(token) {
  return axios.create({
    baseURL: API_URL,
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
}
