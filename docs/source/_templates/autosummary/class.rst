{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autosummary:: {{ objname }}

{% block methods %}
{% if methods %}
.. rubric:: {{ _('Methods') }}

.. autosummary::
{% for item in methods %}
  {%- if not item.startswith('_') and item not in inherited_members %}
  {{ name }}.{{ item }}
  {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}

.. autosummary::
{% for item in attributes %}
  {%- if not item.startswith('_') and item not in inherited_members %}
  {{ name }}.{{ item }}
  {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
