const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs');
const root=require('path').resolve('artifacts/saliency');
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--no-first-run','--disable-background-networking']});
const context=await browser.newContext({viewport:{width:1440,height:1100}});
const page=await context.newPage();let mode='ready';let requests=[];let errors=[];let delay=0,active=0,maxActive=0;let served=0;
page.on('pageerror',e=>errors.push(e.message));
const fixture=JSON.parse(fs.readFileSync(root+'/fixture.json','utf8'));
await page.route('**/*',async route=>{
const url=new URL(route.request().url());requests.push(url.pathname);
if(url.hostname!=='saliency.fixture')return route.abort();
if(url.pathname==='/')return route.fulfill({contentType:'text/html',headers:{'Content-Security-Policy':"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"},body:fs.readFileSync(root+'/fixture.html','utf8')});
if(url.pathname==='/api/saliency'){
active++;maxActive=Math.max(maxActive,active);served++;
const payload=mode==='ready'?fixture:mode==='stream'?{...fixture,frame_id:served,request_id:served}:{state:mode,age_ms:2400};
if(delay)await new Promise(r=>setTimeout(r,delay));
try{return await route.fulfill({contentType:'application/json',body:JSON.stringify(payload)});}finally{active--;}
}
if(url.pathname==='/api/frame.jpg')return route.fulfill({status:204,body:''});
if(url.pathname==='/api/status')return route.fulfill({contentType:'application/json',body:JSON.stringify({status:{mode:'manual',hardware_mode:'simulation',shadow:true,reason:'not_engaged',input_status:'ready'},stale:false,snapshot_age_ms:10,history:[],controls:{}})});
return route.fulfill({status:404,body:''});
});
await page.goto('http://saliency.fixture/');
await page.locator('#saliencyImage').waitFor({state:'visible'});
await page.screenshot({path:root+'/dashboard-desktop.png',fullPage:true});
let dims=await page.locator('#saliencyImage').evaluate(x=>({naturalWidth:x.naturalWidth,naturalHeight:x.naturalHeight}));
if(dims.naturalWidth!==400||dims.naturalHeight!==66)throw Error('wrong composite dimensions');
await page.setViewportSize({width:390,height:1000});await page.waitForTimeout(200);await page.screenshot({path:root+'/dashboard-mobile.png'});await page.locator('#saliencyPanel').screenshot({path:root+'/saliency-mobile-panel.png'});
const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(overflow)throw Error('mobile horizontal overflow');
// A repeated frame should not allocate/decode a new blob every 50 ms.
const repeatedURL=await page.locator('#saliencyImage').getAttribute('src');
await page.waitForTimeout(160);
if(repeatedURL!==await page.locator('#saliencyImage').getAttribute('src'))throw Error('duplicate frame decoded again');
mode='stream';
await page.evaluate(()=>{window.previewFrames=0;window.previewObserver=new MutationObserver(records=>{window.previewFrames+=records.filter(r=>r.attributeName==='src').length;});window.previewObserver.observe(document.getElementById('saliencyImage'),{attributes:true});});
await page.waitForTimeout(1000);
const displayedFrames=await page.evaluate(()=>window.previewFrames);
if(displayedFrames<10)throw Error('presentation remained artificially slow: '+displayedFrames);
// Delayed HTTP responses still have only one request in flight; their age is included.
delay=180;await page.waitForTimeout(800);
if(maxActive!==1)throw Error('overlapping preview requests');
delay=450;await page.waitForTimeout(1000);
if(await page.locator('#saliencyImage').isVisible())throw Error('old in-flight response displayed');
delay=0;mode='stale';await page.waitForFunction(()=>document.getElementById('saliencyBadge').textContent==='STALE');
if(await page.locator('#saliencyImage').isVisible())throw Error('stale image still visible');
await page.screenshot({path:root+'/dashboard-stale.png'});
mode='disabled';await page.waitForFunction(()=>document.getElementById('saliencyBadge').textContent==='OFF');
await page.screenshot({path:root+'/dashboard-disabled.png'});
fs.writeFileSync(root+'/browser-qa.json',JSON.stringify({browser:await browser.version(),errors,dims,displayedFramesPerSecond:displayedFrames,maxConcurrentPreviewRequests:maxActive,mobileOverflow:overflow,states:['ready','stale','disabled'],requests:[...new Set(requests)],scope:'Intercepted synthetic fixture only; no live server or remote device contacted'},null,2));
if(errors.length)throw Error(errors.join('\n'));
await browser.close();console.log('Browser QA passed');
})().catch(e=>{console.error(e);process.exit(1)});
