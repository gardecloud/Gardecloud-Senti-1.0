export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // 1. Basic Security Headers
    const securityHeaders = {
      "Content-Security-Policy": "default-src 'none';",
      "X-Content-Type-Options": "nosniff",
      "X-Frame-Options": "DENY",
      "Senti-Version": "1.0-Stable"
    };

    // 2. Age-Gate Logic
    // We check for a custom header you'll send from your app
    const isAdultVerified = request.headers.get("X-Senti-Age-Verified") === "true";
    const userRole = isAdultVerified ? "UNRESTRICTED" : "RESTRICTED";

    // 3. Health Check (For Monitoring)
    if (url.pathname === "/status") {
      return new Response(JSON.stringify({ 
        status: "Online", 
        mode: userRole,
        engine: "llama.cpp-Senti" 
      }), {
        headers: { ...securityHeaders, "Content-Type": "application/json" }
      });
    }

    // 4. Security Filter (Block simple malicious scripts)
    const bodyText = await request.clone().text();
    const dangerousPatterns = [/eval\(/, /<script/, /sudo\s/];
    if (dangerousPatterns.some(pattern => pattern.test(bodyText))) {
      return new Response("Security Violation: Malicious Pattern Detected", { status: 403 });
    }

    // 5. Proxy logic (Where the magic happens)
    // In a real setup, this would forward the request to your Codespace/Server
    return new Response(`Senti 1.0 (${userRole} Mode) is ready for your input.`, {
      status: 200,
      headers: securityHeaders
    });
  }
};
