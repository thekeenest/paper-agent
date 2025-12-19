# Railway Deployment Guide

This guide will help you deploy the News Origin Finder application to Railway.com.

## Prerequisites

1. A Railway.com account
2. News API key from newsapi.org
3. NewsData API key from newsdata.io
4. OpenAI API key

## Deployment Steps

### 1. Fork the Repository

Fork this repository to your GitHub account.

### 2. Set Up Railway Project

1. Log in to Railway.com with your GitHub account
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your forked repository
5. Railway will detect the Dockerfiles in the repository

### 3. Configure Services

#### Backend Service

1. In your Railway project, select the backend service
2. Go to the "Variables" tab
3. Add the following environment variables:
   - `NEWSAPI_KEY` - Your News API key
   - `NEWSDATA_KEY` - Your NewsData API key
   - `OPENAI_KEY` - Your OpenAI API key
   - `PORT` - Will be set automatically by Railway
   - `CORS_ORIGINS` - The URL of your deployed frontend service
4. Under "Settings", select the following:
   - Root Directory: `/`
   - Dockerfile Path: `backend.Dockerfile`

#### Frontend Service

1. In your Railway project, select the frontend service
2. Go to the "Variables" tab
3. Add the following environment variables:
   - `BACKEND_URL` - The URL of your deployed backend service
4. Under "Settings", select the following:
   - Root Directory: `/`
   - Dockerfile Path: `frontend.Dockerfile`

### 4. Deploy

1. Click "Deploy" for each service
2. Once deployed, Railway will provide URLs for your services
3. You may need to configure a custom domain in Railway settings if desired

## Environment Variables Explanation

### Backend Service

- `NEWSAPI_KEY` - API key for News API
- `NEWSDATA_KEY` - API key for NewsData API
- `OPENAI_KEY` - API key for OpenAI
- `PORT` - Port for the backend service (set automatically by Railway)
- `CORS_ORIGINS` - Comma-separated list of allowed origins for CORS

### Frontend Service

- `BACKEND_URL` - URL of the backend service

## Troubleshooting

If you encounter issues:

1. Check the service logs in Railway dashboard
2. Verify your environment variables are set correctly
3. Ensure your API keys are valid and have sufficient quota
4. Check if the CORS_ORIGINS setting includes your frontend URL

## Local Testing Before Deployment

You can test your application locally using Docker Compose:

```bash
# Build and start the containers
docker-compose up --build

# Access the frontend at http://localhost:8080
```