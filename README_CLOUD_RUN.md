# FUT Europa — despliegue en Google Cloud Run

La aplicación sigue usando Streamlit. Cloud Run ejecuta Streamlit dentro de un contenedor y expone el servicio por HTTPS.

## 1. Requisitos locales

Instala Google Cloud CLI y autentícate:

```bash
gcloud auth login
gcloud auth application-default login
```

Selecciona tu proyecto:

```bash
gcloud config set project TU_PROJECT_ID
```

Habilita las APIs necesarias:

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

## 2. Guardar API_SPORTS_KEY en Secret Manager

No guardes la clave de API-Sports en GitHub.

```bash
printf '%s' 'TU_API_SPORTS_KEY' | gcloud secrets create api-sports-key --data-file=-
```

Si el secreto ya existe, agrega una nueva versión:

```bash
printf '%s' 'TU_API_SPORTS_KEY' | gcloud secrets versions add api-sports-key --data-file=-
```

Obtén la cuenta de servicio usada por Cloud Run y dale permiso para leer el secreto. Si usas la cuenta predeterminada de Compute Engine:

```bash
PROJECT_NUMBER=$(gcloud projects describe TU_PROJECT_ID --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding api-sports-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 3. Desplegar desde el repositorio local

Desde la raíz de `fut_europa`:

```bash
gcloud run deploy fut-europa \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 4 \
  --timeout 900 \
  --set-secrets API_SPORTS_KEY=api-sports-key:latest
```

Cloud Build detectará el `Dockerfile`, construirá la imagen y Cloud Run devolverá la URL HTTPS del servicio.

## 4. Actualizaciones posteriores

Después de hacer cambios en GitHub y actualizar tu copia local:

```bash
git pull

gcloud run deploy fut-europa \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 4 \
  --timeout 900 \
  --set-secrets API_SPORTS_KEY=api-sports-key:latest
```

## 5. Comprobar el servicio

```bash
gcloud run services describe fut-europa \
  --region us-central1 \
  --format='value(status.url)'
```

Logs:

```bash
gcloud run services logs read fut-europa --region us-central1 --limit 100
```

## Notas técnicas

- El contenedor escucha en `0.0.0.0` y usa el puerto que Cloud Run entrega mediante `PORT`.
- Los históricos incluidos en `data/` viajan dentro de la imagen.
- El sistema conserva Streamlit, Elo, ML, Monte Carlo, DuckDB/Parquet y la lógica de apuestas existente.
- El sistema de archivos del contenedor es efímero. No uses archivos locales para guardar datos que deban sobrevivir a nuevas revisiones o reinicios. Para persistencia futura usa Cloud Storage, Firestore, Cloud SQL o Google Sheets según el caso.
- El caché de Streamlit es por instancia de Cloud Run. Una nueva instancia puede tener que reconstruir el caché/modelo en su primera ejecución.
