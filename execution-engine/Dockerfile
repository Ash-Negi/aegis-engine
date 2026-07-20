# Multi-stage: the JDK and the Maven cache stay in the build stage, so the
# shipped image carries a JRE and one jar rather than a toolchain.
FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /build

# Dependencies resolve in their own layer, keyed on the pom alone. Source
# changes then rebuild without re-downloading the world.
COPY pom.xml .
RUN mvn -B dependency:go-offline

COPY src ./src
RUN mvn -B clean package -DskipTests

FROM eclipse-temurin:21-jre-alpine
WORKDIR /app

# wget is what the compose health check calls.
RUN apk add --no-cache wget

# Non-root: the engine needs no privileges, and a compromised container should
# not have any.
RUN addgroup -S aegis && adduser -S aegis -G aegis
USER aegis

COPY --from=build /build/target/execution-engine-*.jar app.jar

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
