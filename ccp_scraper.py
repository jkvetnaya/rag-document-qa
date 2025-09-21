import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
import json

class CCPScraper:
    def __init__(self, base_url="https://codes.findlaw.com/ca/evidence-code/", delay=1):
        self.base_url = base_url
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create directory for saving files
        self.save_dir = "ccp_sections"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def get_section_urls(self, sections):
        """Generate URLs for specific CCP sections"""
        urls = {}
        for section in sections:
            # Clean section number (remove 'Code Civ. Proc., §' prefix if present)
            section_num = re.sub(r'Code Civ\. Proc\.,?\s*§\s*', '', section).strip()
            
            # Handle section ranges like 1220-1222
            if '–' in section_num or '-' in section_num:
                # For ranges, we'll try to get individual sections
                range_match = re.match(r'(\d+)[–-](\d+)', section_num)
                if range_match:
                    start, end = int(range_match.group(1)), int(range_match.group(2))
                    for num in range(start, end + 1):
                        urls[f"CCP_{num}"] = f"{self.base_url}ccp-section-{num}.html"
                continue
            
            # Handle individual sections
            section_num = re.sub(r'[^\d.]', '', section_num)  # Keep only digits and dots
            if section_num:
                urls[f"CCP_{section_num}"] = f"{self.base_url}evid-sect-{section_num}.html"
                
        return urls
    
    def scrape_section(self, url, section_name):
        """Scrape a single CCP section"""
        try:
            print(f"Scraping {section_name}: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to find the main content area
            content_selectors = [
                '.statute-text',
                '.law-text', 
                '.section-content',
                'main',
                '.content',
                '#main-content'
            ]
            
            content = None
            for selector in content_selectors:
                content = soup.select_one(selector)
                if content:
                    break
            
            # If no specific content area found, get the body
            if not content:
                content = soup.find('body')
            
            if content:
                # Extract text and clean it
                text = content.get_text(separator='\n', strip=True)
                
                # Clean up the text
                text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive newlines
                text = re.sub(r'[ \t]+', ' ', text)      # Normalize spaces
                
                return {
                    'section': section_name,
                    'url': url,
                    'text': text,
                    'html': str(content)
                }
            else:
                print(f"Could not find content for {section_name}")
                return None
                
        except requests.RequestException as e:
            print(f"Error scraping {section_name}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error scraping {section_name}: {e}")
            return None
    
    def save_section(self, section_data, format='both'):
        """Save section data to files"""
        if not section_data:
            return
        
        section_name = section_data['section']
        safe_filename = re.sub(r'[^\w.-]', '_', section_name)
        
        if format in ['text', 'both']:
            # Save as text file
            text_file = os.path.join(self.save_dir, f"{safe_filename}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"Section: {section_name}\n")
                f.write(f"URL: {section_data['url']}\n")
                f.write("=" * 50 + "\n\n")
                f.write(section_data['text'])
            print(f"Saved text: {text_file}")
        
        if format in ['html', 'both']:
            # Save as HTML file
            html_file = os.path.join(self.save_dir, f"{safe_filename}.html")
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Section: {section_name} -->\n")
                f.write(f"<!-- URL: {section_data['url']} -->\n")
                f.write(section_data['html'])
            print(f"Saved HTML: {html_file}")
    
    def scrape_sections(self, sections, format='both'):
        """Scrape multiple CCP sections"""
        urls = self.get_section_urls(sections)
        results = []
        
        print(f"Found {len(urls)} sections to scrape")
        
        for section_name, url in urls.items():
            section_data = self.scrape_section(url, section_name)
            if section_data:
                self.save_section(section_data, format)
                results.append(section_data)
            
            # Be respectful - add delay between requests
            time.sleep(self.delay)
        
        # Save summary
        summary_file = os.path.join(self.save_dir, "scraping_summary.json")
        summary = {
            'total_sections': len(urls),
            'successful_scrapes': len(results),
            'sections': [r['section'] for r in results],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nScraping complete!")
        print(f"Successfully scraped: {len(results)}/{len(urls)} sections")
        print(f"Files saved in: {self.save_dir}")
        
        return results

def main():
    # Define the CCP sections from your document
    ccp_sections = [
        "1005(b)",
        "1015", 
        "1016",
        "1209",
        "1209.5", 
        "1218",
        "1218.5",
        "1219",
        "1220–1222",  # This will be expanded to individual sections
        "128(a)(4)",  # Note: This might not follow the same URL pattern
        "1003",
        "415.10",
        "1210",
        "1211", 
        "1217"
    ]
    
    # Initialize scraper
    scraper = CCPScraper(delay=2)  # 2 second delay between requests
    
    # Scrape sections and save both text and HTML
    results = scraper.scrape_sections(ccp_sections, format='both')
    
    # Print summary
    print(f"\nScraping Summary:")
    print(f"Total sections attempted: {len(scraper.get_section_urls(ccp_sections))}")
    print(f"Successful scrapes: {len(results)}")
    
    if results:
        print(f"\nSuccessfully scraped:")
        for result in results:
            print(f"  - {result['section']}")

if __name__ == "__main__":
    main()