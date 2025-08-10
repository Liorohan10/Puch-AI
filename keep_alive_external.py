#!/usr/bin/env python3
"""
External Keep-Alive Service for Render Deployment
This script can be run on any external server/service to keep your Render app active.
"""

import requests
import time
import schedule
import logging
from datetime import datetime
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExternalKeepAlive:
    def __init__(self, target_url: str, ping_interval_minutes: int = 10):
        self.target_url = target_url.rstrip('/')
        self.ping_interval = ping_interval_minutes
        self.success_count = 0
        self.failure_count = 0
        self.last_success = None
        self.last_failure = None
        
    def ping_server(self):
        """Ping the target server to keep it alive"""
        try:
            # Try health check endpoint first
            response = requests.get(
                f"{self.target_url}/health",
                timeout=30,
                headers={
                    'User-Agent': 'External-KeepAlive-Bot/1.0',
                    'Accept': 'application/json'
                }
            )
            
            if response.status_code == 200:
                self.success_count += 1
                self.last_success = datetime.now()
                logger.info(f"✅ Ping successful - Status: {response.status_code}")
                
                # Try to parse response for additional info
                try:
                    data = response.json()
                    if 'timestamp' in data:
                        logger.info(f"   Server timestamp: {data['timestamp']}")
                except:
                    pass
                    
            else:
                logger.warning(f"⚠️ Ping returned non-200 status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.failure_count += 1
            self.last_failure = datetime.now()
            logger.error(f"❌ Ping failed: {str(e)}")
            
            # Fallback: try a simple GET request to root
            try:
                fallback_response = requests.get(
                    self.target_url,
                    timeout=15,
                    headers={'User-Agent': 'External-KeepAlive-Bot/1.0'}
                )
                if fallback_response.status_code < 500:
                    logger.info(f"✅ Fallback ping successful - Status: {fallback_response.status_code}")
                    self.success_count += 1
                    self.last_success = datetime.now()
                else:
                    logger.error(f"❌ Fallback ping also failed - Status: {fallback_response.status_code}")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback ping failed: {str(fallback_error)}")
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure = datetime.now()
            logger.error(f"❌ Unexpected error during ping: {str(e)}")
    
    def get_stats(self):
        """Get statistics about ping attempts"""
        return {
            "target_url": self.target_url,
            "ping_interval_minutes": self.ping_interval,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "uptime_percentage": (self.success_count / max(1, self.success_count + self.failure_count)) * 100
        }
    
    def start_monitoring(self):
        """Start the monitoring loop"""
        logger.info(f"🚀 Starting external keep-alive for {self.target_url}")
        logger.info(f"🔄 Ping interval: {self.ping_interval} minutes")
        
        # Schedule the ping job
        schedule.every(self.ping_interval).minutes.do(self.ping_server)
        
        # Initial ping
        self.ping_server()
        
        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds for scheduled jobs
                
                # Log stats every hour
                if self.success_count + self.failure_count > 0 and (self.success_count + self.failure_count) % 6 == 0:
                    stats = self.get_stats()
                    logger.info(f"📊 Stats: {stats['success_count']} successes, {stats['failure_count']} failures, {stats['uptime_percentage']:.1f}% uptime")
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Keep-alive monitoring stopped by user")
            stats = self.get_stats()
            logger.info(f"📊 Final stats: {json.dumps(stats, indent=2)}")

def main():
    # Configuration
    TARGET_URL = os.getenv("TARGET_URL", "https://puch-ai-ssnl.onrender.com")
    PING_INTERVAL = int(os.getenv("PING_INTERVAL_MINUTES", "10"))
    
    # Start keep-alive service
    keep_alive = ExternalKeepAlive(TARGET_URL, PING_INTERVAL)
    keep_alive.start_monitoring()

if __name__ == "__main__":
    main()
