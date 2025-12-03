import Fastify from 'fastify';

// 🚀 Initialize Fastify server
const fastify = Fastify({
  logger: true
});

// 🌐 Define /test route that returns hello world JSON
fastify.get('/test', async (request, reply) => {
  return { message: 'hello world' };
});

// 🔌 Start the server
const startServer = async () => {
  try {
    await fastify.listen({ port: 3000, host: '0.0.0.0' });
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

startServer();
