import asyncio
import sys
from datetime import datetime
from volcenginesdkarkruntime import AsyncArk
from gpt.batch_api import generate_batch_payload, process_batch_async

async def test_ark_batch_processing():
    start = datetime.now()
    # 准备测试请求
    test_requests = [{
        "messages": [
            {"role": "system", "content": "You are a marketing assistant scoring Reddit posts for product relevance."},
            {"role": "user", "content": '''
             You are a product research assistant.  
You will receive a Reddit post along with up to 10 top comments.  
Your job is to identify if the discussion reveals a user need, frustration, or wish related to a tool, service, or website.  
**Guidelines:**
- Combine insights from both post and comments to find the clearest user need(s).  
Analyze:
- relevance: Does the post mention problems, missing features, or desired improvements related to software, tools, or websites?
- pain_point: What is the specific difficulty, limitation, or unmet need?
- How emotionally charged is the post (neutral, frustrated, urgent, etc.)?

Your output should include:
- `relevance_score` (0–10)
- `emotional_intensity` (0–10)
- `pain_point_clarity` (0–10)
- `summary`: a 1-sentence summary of the user’s problem or wish

Respond with a JSON object like:
{
  "relevance_score": 8,
  "emotional_intensity": 6,
  "pain_point_clarity": 9,
  "summary": "One-sentence summary of the core user need or frustration."
}
My first startup failed in 2019 after burning through our pre-seed and nearly two years of mine and my cofounder's time. Learned a valuable lesson of 'if you build it.. no one will come'.

Second startup failed after a year due to covid - lesson there: merging online and offline is hard, unit economy MUST make sense and can withstand shocks and bad time.

Third startup failed after two and half years (nearly full length of covid), lesson there was: it's much better to be early and suck than be late and mediocre.

Fourth startup was more of an agency, took a break from chasing VC money and decided to offer a service and make money, that spanned a year and half from 2023 to mid 2024. Lesson there: much better to offer service first then build a product no one want to use.

Now, on to the fifth startup, 6 months in 25K MRR over 80% margin while we offer a mix of service (done for you) and product combo. Lesson here so far: offer service and hire smart people to gain momentum and move fast.
 comment:I can relate to this rollercoaster, I am now on my third startup, but the path here looks a lot like your journey.

My first one launched in 2013, an India focused e-commerce play from day one. We were chasing desktop customers just as the market started swinging hard to mobile. Our runway was short, but instead of pivoting fast, we kept patching the old approach. That stubbornness hurt. Good revenue, learning and profits. Took an exit on time.

Second venture was in the SaaS space, global from day 1, and I thought I learned my lesson, until growth brought in the wrong hires, built and run for 8 years. We gave the team four chances over four years to rinse and shine, changing leadership and structure each time, but the drain on the founders was brutal. Eventually, we scaled the decision down after 8 years. It was profitable and gave good experience in US and EU markets.

Now with Sprout24, I have baked those scars into the model, no early large team hires in in India/Asia, focus on profitable product from day one with AI at core, and move only when the market is signaling readiness. It is less romantic than “build it and they will come,” but it is a lot more fun when the cash flow works.
 comment:Starting a business can be like starting to work out or wanting to lose weight. 

People love the IDEA of the end result and will buy running shoes, sign up to a gym, create a meal plan and buy all these apps for tracking, then run out of time or motivation to actually do any exercise. They’ve confused all of this prep work, design and purchases as actually improving their health. 

The first step when starting a low risk business is MAKE MONEY. Sale your service, deliver the service, create revenue and then setup processes around them, build the app, hire your first employee, whatever.
 comment:Great share OP 🙏

I once heard a similar story told and it was described as tuition payments.  
After your four years of university you are paid up, graduated and ready to go. 

The world has gone so SaaS crazy that they have lost site on the value of services.  Paid R&D, cashflow, direct connect to customer pain translated to products with value and utility… 

Good luck!
 comment:Interesting!! How do people usually manage their personal expenses in the given situations? Like you withdraw a salary from each startup? Or burn your savings for it? 

I do have an idea id like to pursue but everytime it comes to my personal expenses and how’d I manage that? Any insights on that?
 comment:25K MRR with 80% margin in 6 months is solid traction. If you keep that service+product mix balanced, you might be able to reinvest enough to avoid VC entirely, which could preserve flexibility long term.
 comment:The honest journey through five startups because it shows how each attempt adds to the playbook i can relate to the lesson about offering a service first to validate demand before building a product as it keeps cash flow alive while refining what people will actually pay for combining high margin services with a scalable product and a smart team seems like a strong model if you are open i would be happy to share some simple ai automation workflows that can help speed up service delivery and improve margins feel free to dm anytime
 comment:I've never gone the VC route. How were you able to keep attaining funding so rapidly after repeated failures? Also, this post is really nice. So many founders exaggerate and puff their chest up and give the ol' swagger. I love honest real depictions of what our world is actually like.  I also love the tenacity stories. The "so many initial failures" stories always seem to end in a stunning success. I wish that for you!
 comment:Thanks for sharing “hire smart people” usually this solves a lot of problems. Varsity teams win big games
 comment:I think you need to Focus on your 5th Startup
 comment:Respect for honesty. It seems you treat your start up like jobs and you just change it when it doesnt work directly. Why arent you pushing through with an idea and doing it longer? Or did you always burn through all the cash? Friends startet an agency change services quite a bit over time failed sth a little, but after 8 years have now 30 employees big margin and a brand. Some customer know them for 6 years. IT Partner know them under the brand name. They are a small consultant firm rather than just a startup which builds Trust in customer.
'''}
        ]
    } for _ in range(5)]

    # 生成AsyncArk请求
    ark_requests = generate_batch_payload(test_requests, model="deepseek-v3-1-250821")
    results, errors = await process_batch_async(ark_requests, model="deepseek-v3-1-250821")

    end = datetime.now()
    print(f"Total time: {end - start}, Success: {len(results)}, Errors: {len(errors)}")
    for result in results[:3]:
        print(f"Result: {result['response']['choices'][0]['message']['content'][:50]}...")

if __name__ == "__main__":
    asyncio.run(test_ark_batch_processing())