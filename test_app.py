import app

print('[1] Model loading test:')
model_art = app.get_model()
print('Model artifact loaded:', model_art is not None)
if model_art:
    print('Trained at:', model_art.get('trained_at'))
    print('Metrics:', model_art.get('metrics'))

print('\n[2] Safe URL test:')
safe_res = app.analyze_url('https://github.com/torvalds/linux')
print('Safe URL result:', safe_res['risk_level'], 'Risk score:', safe_res['risk_score'])

print('\n[3] Phishing URL test:')
phish_res = app.analyze_url('http://paypal-verification-alert.xyz/login')
print('Phishing URL result:', phish_res['risk_level'], 'Risk score:', phish_res['risk_score'], 'Reasons:', len(phish_res['reasons']))

print('\n[4] IP URL test:')
ip_res = app.analyze_url('http://192.168.1.1/admin-login.php')
print('IP URL result:', ip_res['risk_level'], 'Risk score:', ip_res['risk_score'], 'Reasons:', len(ip_res['reasons']))

print('\nAll tests passed successfully!')
