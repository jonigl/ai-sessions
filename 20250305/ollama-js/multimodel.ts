import ollama from 'ollama'

async function main() {
  const imagePath = './img/image.png'
  const response = await ollama.generate({
    model: 'llava:13b',
    prompt: 'describe this image:',
    images: [imagePath],
    stream: true,
  })
  for await (const part of response) {
    process.stdout.write(part.response)
  }
}

main().catch(console.error)
