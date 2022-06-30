# Docker-Compose

It is very useful for local development to startup one/multiple service in a clean environment.

Pure docker can help to a certain level. But docker-compose is extremely useful if there is a long list of arguments to startup a docker container, e.g. binding ports, mounting volumes…etc

Some more advanced usage such as networking, managing existed container, ENV, …etc

## Common usage

### Basic example

```yaml
version: '3.8'
services:
  app:
    build: .
			# Use below setting if dockerfile name is changed
      # context: .
      # dockerfile: <specific dockerfile name>
    volumes:
      - ./src:/app
    ports:
      - "3000:3000"
```