from fastapi import Header, HTTPException, status
from typing import Optional
import time

# Simple rate limiting
request_times = {}

def rate_limit(api_key: Optional[str] = Header(None), 
               max_requests: int = 100,
               time_window: int = 3600):
    """Simple rate limiting middleware"""
    
    if api_key is None:
        # For demo, allow without API key
        return
    
    current_time = time.time()
    
    if api_key not in request_times:
        request_times[api_key] = []
    
    # Clean old requests
    request_times[api_key] = [
        req_time for req_time in request_times[api_key]
        if current_time - req_time < time_window
    ]
    
    # Check rate limit
    if len(request_times[api_key]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Add current request
    request_times[api_key].append(current_time)
    
    return api_key

def verify_api_key(api_key: Optional[str] = Header(None)):
    """Verify API key (demo version - always returns True)"""
    # In production, validate against database
    if api_key and api_key.startswith("demo_"):
        return api_key
    
    # For demo purposes, allow all requests
    return "demo_key"