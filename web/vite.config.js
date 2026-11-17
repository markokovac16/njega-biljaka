import { fileURLToPath } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

const mapaKljuca = fileURLToPath(new URL('..', import.meta.url))

export default defineConfig(({ command, mode }) => {
  const env = command === 'serve' ? loadEnv(mode, mapaKljuca, 'CLAUDE_') : {}
  return {
    plugins: [vue()],
    server: {
      fs: { allow: [fileURLToPath(new URL('.', import.meta.url)), fileURLToPath(new URL('../znanje', import.meta.url))] },
    },
    define: {
      __KLJUC__: JSON.stringify(env.CLAUDE_API_KEY ?? ''),
      __MODEL__: JSON.stringify(env.CLAUDE_MODEL || 'claude-haiku-5-5'),
    },
  }
})
