type Environment = {
	readonly IMAGE_QUEUE: Queue;
  };
  

export default {

	// Write an endpoint which calls the correct model service
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {

		let UUID = crypto.randomUUID();
		const formdata = await request.formData();
		const image = formdata.get('image');

		var payload = {
			"UUID":UUID,
			"image": image
		}

		const url = new URL(request.url)
		const pathname = url.pathname

		if (pathname.startsWith('/fetch')) {
			
			// Get UUID from form data
			const value = await env.AIMICRO.get(formdata.get('UUID'),{ cacheTtl: 60 });

            if (value === null) {
                return new Response("ID not found or not yet processed", {status: 404});
            }

            return new Response(value);

		} else if (pathname.startsWith('/inference')){
			
			// Send imge to model evaluation queue
			await env.IMAGE_QUEUE.send(payload, {'content-type': 'json'});
			// Return a UUID to lookup in KV
			return new Response(UUID);	
		}
	},
};
