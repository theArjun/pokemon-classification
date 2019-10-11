from django import forms


class ClassifyForm(forms.Form):
    url = forms.URLField(required=False,
                         widget=forms.URLInput(
                             attrs={
                                 'class': 'form-control',
                                 'placeholder': 'Paste Image URL (optional)'
                             }
                         ))

    img = forms.FileField(required=False,
                            widget=forms.FileInput(
                                attrs={
                                    'class': 'form-control',
                                }
                            ))

    # def clean(self):
    #
    #     url = self.cleaned_data.get('url')
    #     image = self.cleaned_data.get('image')
    #
    #     if url or image:
    #         return
    #     raise forms.ValidationError('Any one of them should be filled.')

